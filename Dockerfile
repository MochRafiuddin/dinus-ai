# Menggunakan base image Python yang stabil
FROM python:3.10-slim

# Install library sistem terbaru yang dibutuhkan oleh OpenCV & PyTorch di Debian Trixie
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install paket Python yang dibutuhkan
RUN pip install --no-cache-dir runpod ultralytics requests

# Menyalin seluruh file project (termasuk app.py dan model .pt) ke dalam container
COPY . .

# Jalankan skrip test load model di awal saat build agar memastikan file .pt valid
RUN python -c "from ultralytics import YOLO; YOLO('model_chompchomp_new.pt'); YOLO('model_beutycharm_new.pt')"

CMD [ "python", "-u", "app.py" ]