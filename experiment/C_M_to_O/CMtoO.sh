
echo "=== BẮT ĐẦU CHUẨN BỊ MÔI TRƯỜNG HUẤN LUYỆN ==="

# 1. Bê file config.py hiện tại ra thư mục gốc
cp config.py ../../config.py
echo "Đã nạp file config.py ra thư mục gốc thành công!"

# 2. Khai hỏa Teacher Model
echo "Đang khởi động AI... Quá trình huấn luyện bắt đầu!"
python train_ssdg_full.py
