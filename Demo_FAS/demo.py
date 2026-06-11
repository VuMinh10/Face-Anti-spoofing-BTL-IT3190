import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.StudentNet import SSDG_MobileNet

# 1. Khai báo danh sách các file mô hình
MODEL_PATHS = {
    "Test MSU": "./weights/msu_model_best_0.07559_113.pth.tar",
    "Test OULU": "./weights/oulu_model_best_0.15244_118.pth.tar",     
    "Test CASIA": "./weights/casia_model_best_0.09219_106.pth.tar"   
}

# 2. Khởi tạo mạng AI rỗng 
device = torch.device("cpu") 
model = SSDG_MobileNet(num_classes=2, num_domains=3).to(device)
current_loaded_model = None  # Biến theo dõi xem model nào đang được lắp

# Hàm nạp model
def load_weights(model_choice):
    global current_loaded_model
    # Chỉ load lại file nếu chọn mô hình khác
    if current_loaded_model != model_choice:
        print(f"[*] Đang chuyển sang mô hình: {model_choice}...")
        path = MODEL_PATHS[model_choice]
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        current_loaded_model = model_choice

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Hàm xử lý ảnh từ giao diện 
def predict_fas(img, model_choice):
    if img is None:
        return {"Lỗi: Vui lòng tải ảnh lên hoặc bật Webcam chụp trước.": 1.0}
    
    try:
        load_weights(model_choice)
    except Exception as e:
        return {f"Lỗi không tìm thấy file trọng số: {str(e)}": 1.0}

    # Tiền xử lý
    img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Dự đoán
    with torch.no_grad():
        logits, _ = model(img_tensor, norm_flag=True)
        probs = F.softmax(logits, dim=1)[0]
    
    # Trả về kết quả
    return {"Ảnh Giả Mạo (Spoof)": float(probs[0]), "Người Thật (Real)": float(probs[1])}

# 4. Tạo giao diện Web
demo = gr.Interface(
    fn=predict_fas,
    inputs=[
        gr.Image(type="pil", label="Tải ảnh lên hoặc Bật Webcam chụp"),
        gr.Dropdown(choices=list(MODEL_PATHS.keys()), value="Kịch bản 1: Học sinh Test MSU", label="Chọn Kịch bản (Mô hình AI)")
    ],
    outputs=gr.Label(num_top_classes=2, label="Kết quả chẩn đoán"),
    title="Hệ thống Phát hiện Giả mạo Khuôn mặt (FAS)",
    description="Sử dụng kỹ thuật Domain Generalization & Knowledge Distillation (MobileNetV2)\n\n*Lưu ý: Chụp cận mặt trong môi trường đủ sáng để AI phân tích tốt nhất.*"
)

if __name__ == "__main__":
    demo.launch()