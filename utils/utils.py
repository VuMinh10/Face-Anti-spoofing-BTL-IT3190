import json
import math
import pandas as pd
import torch
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]
        i += 1
        if(epoch <= lr_epoch_1):
            param_group['lr'] = init_lr * 0.1 ** 0
        elif(epoch <= lr_epoch_2):
            param_group['lr'] = init_lr * 0.1 ** 1
        else:
            param_group['lr'] = init_lr * 0.1 ** 2

def draw_roc(frr_list, far_list, roc_auc):
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.title('ROC')
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [1, 0], 'r--')
    plt.grid(ls='--')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    save_dir = './save_results/ROC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('./save_results/ROC/ROC.png')
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')
    save_json = []
    dict = {}
    dict['FAR'] = far_list
    dict['FRR'] = frr_list
    save_json.append(dict)
    json.dump(save_json, file, indent=4)

def sample_frames(flag, num_frames, dataset_name):
    '''
    Hàm tự động gom nhóm ảnh theo Video và lấy mẫu khung hình chuẩn xác.
    '''
    label_dir = os.path.join('/kaggle/working/BTL-IT3190/data_label', dataset_name)
    
    if flag == 0:
        label_path = os.path.join(label_dir, 'fake_label.json')
        save_label_path = os.path.join(label_dir, 'choose_fake_label.json')
    elif flag == 1:
        label_path = os.path.join(label_dir, 'real_label.json')
        save_label_path = os.path.join(label_dir, 'choose_real_label.json')
    else:
        label_path = os.path.join(label_dir, 'all_label.json')
        save_label_path = os.path.join(label_dir, 'choose_all_label.json')

    with open(label_path, 'r') as f:
        all_label_json = json.load(f)

    from collections import defaultdict
    video_groups = defaultdict(list)

    for item in all_label_json:
        photo_path = item['photo_path']
        filename_no_ext = os.path.splitext(os.path.basename(photo_path))[0]
        
        if dataset_name == 'oulu':
            parts = filename_no_ext.split('_')
            video_name = '_'.join(parts[:-1]) if len(parts) > 1 else filename_no_ext
        else:
            # MSU hoặc CASIA (Dùng thư mục làm tên video nếu tên file là frame_...)
            if 'frame' in filename_no_ext.lower() or filename_no_ext.isdigit():
                video_name = os.path.basename(os.path.dirname(photo_path))
            else:
                parts = filename_no_ext.split('_')
                video_name = '_'.join(parts[:-1]) if len(parts) > 1 else filename_no_ext
                
        video_groups[video_name].append(item)

    final_json = []
    video_id = 0

    for v_name, frames in video_groups.items():
        frames = sorted(frames, key=lambda x: x['photo_path'])
        if len(frames) > 0:
            indices = np.linspace(0, len(frames) - 1, min(num_frames, len(frames)), dtype=int)
            for idx in indices:
                dict_item = frames[idx].copy()
                dict_item['photo_belong_to_video_ID'] = video_id
                final_json.append(dict_item)
        video_id += 1

    if flag == 0:
        print("Total video number(fake): ", video_id, dataset_name)
    elif flag == 1:
        print("Total video number(real): ", video_id, dataset_name)
    else:
        print("Total video number(target): ", video_id, dataset_name)

    with open(save_label_path, 'w') as f_sample:
        json.dump(final_json, f_sample, indent=4)

    return pd.read_json(save_label_path)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mkdirs(checkpoint_path, best_model_path, logs):
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path): os.makedirs(best_model_path)
    if not os.path.exists(logs): os.mkdir(logs)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None
    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()
    def flush(self): pass

def save_checkpoint(save_list, is_best, model, gpus, checkpoint_path, best_model_path, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_ACER = save_list[4]
    threshold = save_list[5]
    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1): k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state = {
            "epoch": epoch, "state_dict": new_state_dict, "valid_arg": valid_args,
            "best_model_EER": best_model_HTER, "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC, "threshold": threshold
        }
    else:
        state = {
            "epoch": epoch, "state_dict": model.state_dict(), "valid_arg": valid_args,
            "best_model_EER": best_model_HTER, "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC, "threshold": threshold
        }
    filepath = checkpoint_path + filename
    torch.save(state, filepath)
    if is_best:
        shutil.copy(filepath, best_model_path + 'model_best_' + str(best_model_HTER) + '_' + str(epoch) + '.pth.tar')

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
