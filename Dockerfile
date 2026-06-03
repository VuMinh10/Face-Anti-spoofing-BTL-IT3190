# Sử dụng base image hiện đại của PyTorch (đã gói sẵn Python 3.10, PyTorch 2.1, CUDA 11.8)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Tránh bị kẹt giao diện chờ nhập Y/N khi cài đặt thư viện hệ thống
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt một số công cụ hệ thống cần thiết (thường dùng cho các thư viện xử lý ảnh CV2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /workspace/SSDG

# (Tùy chọn) Copy và cài đặt requirements nếu nhóm bạn có sẵn file này
# COPY requirements.txt /workspace/SSDG/
# RUN pip install -r requirements.txt