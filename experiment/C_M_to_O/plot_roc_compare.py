import sys
sys.path.append('../../') 

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os

from config import config
from models.DGFAS import DG_model 
from models.StudentNet import SSDG_MobileNet 
from utils.get_loader import get_dataset

def get_predictions(model, data_loader, device):
    """Hàm chạy mô hình để lấy điểm dự đoán và nhãn thực tế"""
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(device)
            labels = batch[1]
            
            # Gọi model, trả về logits và feature
            out1, out2 = model(inputs, config.norm_flag)
            
            # Lọc Logits (kích thước = 2)
            logits = out1 if out1.size(1) == 2 else out2
            
            # Dùng Softmax để chuyển Logits thành Xác suất (0 -> 1)
            probs = F.softmax(logits, dim=1)
            
            # Lấy xác suất của nhãn "Thật" (Giả sử nhãn 1 là Thật)
            scores = probs[:, 1].cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_scores)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("1. Đang nạp tập dữ liệu thi (Target Domain)...")
    _, _, _, _, tgt_valid_loader = get_dataset(
        config.src1_data, config.src1_train_num_frames,
        config.src2_data, config.src2_train_num_frames,
        config.tgt_data, config.tgt_test_num_frames, config.batch_size
    )

    print("2. Đang nạp Thầy Giáo (ResNet18)...")
    teacher = DG_model(config.model).to(device)
    teacher_path = '/kaggle/input/datasets/vuminh10/fas-teacher-weights/model_best_0.15143_24.pth.tar'
    teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=False)['state_dict'])
    
    print("3. Đang nạp Học Sinh (MobileNetV2)...")
    student = SSDG_MobileNet(num_classes=2, num_domains=3).to(device)
    student_path = '/kaggle/input/datasets/vuminh10/student-model/model_best_0.15244_118.pth.tar'
    checkpoint = torch.load(student_path, map_location=device, weights_only=False)
    student.load_state_dict(checkpoint['state_dict'])

    print("4. Cho Thầy và Trò làm bài thi...")
    y_true_t, y_scores_t = get_predictions(teacher, tgt_valid_loader, device)
    y_true_s, y_scores_s = get_predictions(student, tgt_valid_loader, device)

    print("5. Đang vẽ biểu đồ ROC...")
    # Tính toán ROC cho Thầy
    fpr_t, tpr_t, _ = roc_curve(y_true_t, y_scores_t)
    roc_auc_t = auc(fpr_t, tpr_t)

    # Tính toán ROC cho Trò
    fpr_s, tpr_s, _ = roc_curve(y_true_s, y_scores_s)
    roc_auc_s = auc(fpr_s, tpr_s)

    # Vẽ hình
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_t, tpr_t, color='blue', linestyle='--', lw=2, label=f'Teacher ResNet18 (AUC = {roc_auc_t:.3f})')
    plt.plot(fpr_s, tpr_s, color='red', linestyle='-', lw=2, label=f'Student MobileNetV2 (AUC = {roc_auc_s:.3f})')
    
    # Đường chéo cơ sở (Đoán bừa 50%)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Bắt nhầm người thật)')
    plt.ylabel('True Positive Rate (Chặn đúng kẻ gian)')
    plt.title('Biểu đồ ROC so sánh Teacher vs Student trên OULU')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = 'ROC_Comparison_OULU.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Đã vẽ xong. Ảnh được lưu tại: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    main()
