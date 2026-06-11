import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class SSDG_MobileNet(nn.Module):
    def __init__(self, num_classes=2, num_domains=3, pretrained=True):
        """
        Args:
            num_classes: Số lượng nhãn (2: Real/Fake)
            num_domains: Số lượng domain (Ví dụ: 3 cho Oulu, Casia, MSU)
            pretrained: Có dùng weight ImageNet không
        """
        super(SSDG_MobileNet, self).__init__()
        
        # 1. Load Backbone MobileNetV2
        # MobileNetV2 đầu ra feature map cuối cùng có 1280 channels
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Lấy phần features extractor (bỏ phần classifier gốc)
        self.features = backbone.features
        
        # 2. Bottleneck Layer
        # Giảm chiều từ 1280 xuống 512 (để giống kích thước feature của ResNet-18 trong SSDG)
        # Giúp tính toán Triplet Loss nhẹ hơn
        self.bottleneck = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 3. Class Classifier (Real vs Fake)
        self.classifier = nn.Linear(512, num_classes)
        
        # 4. Domain Classifier (Phân biệt nguồn dữ liệu - Cho Domain Generalization)
        # Trong code train distillation bạn có thể không dùng cái này tính loss, 
        # nhưng vẫn cần khai báo để khớp format đầu ra.
        self.domain_classifier = nn.Linear(512, num_domains)

        # Khởi tạo weights cho các lớp mới thêm vào
        self._initialize_weights()

    def forward(self, x, norm_flag=True): # <--- Thêm tham số norm_flag
        # --- Feature Extraction ---
        x = self.features(x) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) 
        
        # --- Bottleneck ---
        feat = self.bottleneck(x) 
        
        # Chuẩn hóa L2 cho đặc trưng để tính Triplet Loss mượt hơn
        if norm_flag:
            feature_norm = torch.norm(feat, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feat = torch.div(feat, feature_norm)
            
        # --- Classification Heads ---
        class_logits = self.classifier(feat)      
        domain_logits = self.domain_classifier(feat) 
        
        return class_logits, feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    # Test thử model để đảm bảo không lỗi dimension
    x = torch.randn(8, 3, 256, 256)
    model = SSDG_MobileNet(num_classes=2, num_domains=3)
    feat, cls, dom, _ = model(x)
    print("Feature shape:", feat.shape) # Mong đợi: [8, 512]
    print("Class logits:", cls.shape)   # Mong đợi: [8, 2]
    print("Domain logits:", dom.shape)  # Mong đợi: [8, 3]
    print("Model initialized successfully!")