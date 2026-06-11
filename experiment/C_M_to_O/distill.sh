echo "=== Khởi động chưng cất trên OULU ==="
cp config.py ../../config.py
mkdir -p logs
python train_ssdg_distill.py
