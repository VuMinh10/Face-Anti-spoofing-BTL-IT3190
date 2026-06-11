import sys
sys.path.append('../../') 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random
import time
from timeit import default_timer as timer

from config import config
from models.DGFAS import DG_model 
from models.StudentNet import SSDG_MobileNet 
from utils.utils import save_checkpoint, Logger, time_to_str, adjust_learning_rate
from utils.get_loader import get_dataset
from utils.evaluate import eval
from loss.hard_triplet_loss import HardTripletLoss

def loss_fn_kd(outputs, teacher_outputs, T=4.0):
    KD_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1)
    ) * (T * T)
    return KD_loss

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Seed & Môi trường
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    student_checkpoint_path = config.checkpoint_path.replace('resnet18', 'mobilenet_distill')
    student_best_path = config.best_model_path.replace('resnet18', 'mobilenet_distill')
    if not os.path.exists(student_checkpoint_path): os.makedirs(student_checkpoint_path)
    if not os.path.exists(student_best_path): os.makedirs(student_best_path)

    # 2. Chuẩn bị data loaders
    print("==> Loading Datasets...")
    src1_loader_fake, src1_loader_real, src2_loader_fake, src2_loader_real, tgt_valid_loader = get_dataset(
        config.src1_data, config.src1_train_num_frames,
        config.src2_data, config.src2_train_num_frames,
        config.tgt_data, config.tgt_test_num_frames, config.batch_size
    )

    # 3. Khởi tạo models
    print("==> Loading Teacher Model (ResNet18)...")
    net_teacher = DG_model(config.model).to(device)
    
    teacher_path = '/kaggle/input/datasets/vuminh10/fas-teacher-weights/model_best_0.16975_98.pth.tar'
    
    if os.path.isfile(teacher_path):
        teacher_checkpoint = torch.load(teacher_path, weights_only=False)
        net_teacher.load_state_dict(teacher_checkpoint['state_dict'])
        print(f"--> Đã nạp Teacher thành công!")
    else:
        print(f"LỖI: Không tìm thấy Teacher tại {teacher_path}")
        return

    net_teacher.eval()
    for param in net_teacher.parameters(): param.requires_grad = False

    print("==> Initializing Student Model (MobileNetV2)...")
    net_student = SSDG_MobileNet(num_classes=2, num_domains=3, pretrained=True).to(device)

    # 4. OPTIMIZER, LOSS & HYPERPARAMETERS
    init_lr = 0.005 # Khởi điểm thấp hơn Teacher một chút
    optimizer = optim.SGD(net_student.parameters(), lr=init_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    init_param_lr = [group["lr"] for group in optimizer.param_groups]
    
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_triplet = HardTripletLoss(margin=0.1, hardest=False).to(device)
    criterion_mse = nn.MSELoss().to(device) # Để học Feature map
    
    T = 4.0   
    alpha = 0.5  # Trọng số cho KD Logits
    beta = 50.0  # Trọng số cho KD Feature 

    # 5. Cài đặt bảng log 
    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_Distill.txt', mode='a')
    log.write('  iter  |  Cls_L   KD_L   Feat_L |   top-1   HTER    AUC    |    time      |\n')
    log.write('-----------------------------------------------------------------------------|\n')

    print("==> Start Training Knowledge Distillation...")
    
    iter_per_epoch = 10
    best_model_HTER = 1.0
    best_model_ACC = 0.0
    best_model_AUC = 0.0
    start_time = timer()
    epoch = 1

    # Tạo iterators
    iter_s1_f, iter_s1_r = iter(src1_loader_fake), iter(src1_loader_real)
    iter_s2_f, iter_s2_r = iter(src2_loader_fake), iter(src2_loader_real)

    # Vòng lặp Iteration chuẩn của SSDG 
    for iter_num in range(config.max_iter + 1):
        if (iter_num % len(src1_loader_fake) == 0): iter_s1_f = iter(src1_loader_fake)
        if (iter_num % len(src1_loader_real) == 0): iter_s1_r = iter(src1_loader_real)
        if (iter_num % len(src2_loader_fake) == 0): iter_s2_f = iter(src2_loader_fake)
        if (iter_num % len(src2_loader_real) == 0): iter_s2_r = iter(src2_loader_real)

        if (iter_num != 0 and iter_num % iter_per_epoch == 0): epoch += 1

        net_student.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)

        data_s1_f, label_s1_f = next(iter_s1_f)
        data_s1_r, label_s1_r = next(iter_s1_r)
        data_s2_f, label_s2_f = next(iter_s2_f)
        data_s2_r, label_s2_r = next(iter_s2_r)

        data = torch.cat((data_s1_r, data_s1_f, data_s2_r, data_s2_f), 0).to(device)
        labels = torch.cat((label_s1_r, label_s1_f, label_s2_r, label_s2_f), 0).to(device)

        # Tạo Domain Labels cho Triplet Loss
        r_dom1 = torch.LongTensor(data_s1_r.shape[0], 1).fill_(0).to(device)
        r_dom2 = torch.LongTensor(data_s2_r.shape[0], 1).fill_(0).to(device)
        f_dom1 = torch.LongTensor(data_s1_f.shape[0], 1).fill_(1).to(device)
        f_dom2 = torch.LongTensor(data_s2_f.shape[0], 1).fill_(2).to(device)
        domain_labels = torch.cat([r_dom1, f_dom1, r_dom2, f_dom2], dim=0).view(-1)

        # --- Bước 1: Forward ---
        with torch.no_grad():
            t_out1, t_out2 = net_teacher(data, config.norm_flag)
            if t_out1.size(1) == 2:
                teacher_logits, feat_teacher = t_out1, t_out2
            else:
                feat_teacher, teacher_logits = t_out1, t_out2
        
        s_out1, s_out2 = net_student(data, config.norm_flag)
        if s_out1.size(1) == 2:
            student_logits, feat_student = s_out1, s_out2
        else:
            feat_student, student_logits = s_out1, s_out2

        # --- Bước 2: Tính Loss toàn diện ---
        # A. Hard Loss & Triplet (Học tự lập)
        loss_cls = criterion_cls(student_logits, labels)
        loss_triplet = criterion_triplet(feat_student, domain_labels)
        loss_hard = loss_cls + (config.lambda_triplet * loss_triplet)

        # B. Soft Loss (Học Logits từ Teacher)
        loss_soft = loss_fn_kd(student_logits, teacher_logits, T)

        # C. Feature Loss (Học nếp nhăn không gian từ Teacher)
        loss_feat = criterion_mse(feat_student, feat_teacher)

        # Tổng Loss
        total_loss = (1 - alpha) * loss_hard + alpha * loss_soft + beta * loss_feat

        # --- Bước 3: Backward ---
        total_loss.backward()
        optimizer.step()

        print(f'\r Iter: {iter_num}/{config.max_iter} | TotL: {total_loss.item():.3f} (Cls: {loss_cls.item():.3f}, Soft: {loss_soft.item():.3f}, Feat: {loss_feat.item():.3f})', end='', flush=True)

        # --- BƯỚC 4: Evualtion & Lưu checkpoint (Theo mốc iter_per_epoch) ---
        if (iter_num != 0 and (iter_num + 1) % iter_per_epoch == 0):
            valid_args = eval(tgt_valid_loader, net_student, config.norm_flag)
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            
            if is_best:
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, 0, valid_args[5]]
            save_checkpoint(save_list, is_best, net_student, config.gpus, student_checkpoint_path, student_best_path)
            
            print('\r', end='', flush=True)
            log.write('  %4.1f  |  %5.3f  %5.3f  %5.3f |  %6.3f  %6.3f  %6.3f  | %s\n'
                      % ((iter_num + 1) / iter_per_epoch, loss_cls.item(), loss_soft.item(), loss_feat.item(),
                         float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                         time_to_str(timer() - start_time, 'min')))
            time.sleep(0.01)

if __name__ == '__main__':
    main()
