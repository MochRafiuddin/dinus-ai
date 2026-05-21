import runpod
import requests
from ultralytics import YOLO
import os
import datetime
import cv2
import base64

# Me-load model custom milikmu yang sudah tertanam di image
models = {
    "chompchomp": YOLO("model_chompchomp_new.pt"),
    "beutycharm": YOLO("model_beutycharm_new.pt")
}

def group_boxes_into_rows(boxes, threshold=100):
    try:
        rows = []
        boxes.sort(key=lambda box: box['box'][1])  # Sort by the top-left y-coordinate

        for box in boxes:
            added_to_row = False
            for row in rows:
                if abs(box['box'][1] - row[0]['box'][1]) < threshold:
                    row.append(box)
                    added_to_row = True
                    break
            if not added_to_row:
                rows.append([box])
        
        # Sort each row horizontally by the top-left x-coordinate
        for row in rows:
            row.sort(key=lambda box: box['box'][0])

        return rows
    except Exception as e:
        print(f"Error in group_boxes_into_rows: {e}")
        return []

def handler(job):
    start_time = datetime.datetime.now().timestamp()
    job_input = job.get('input', {})
    
    # Payload baru: list of objects [{"url": "...", "type": "chompchomp"}]
    images_input = job_input.get('images', [])
    
    # Fallback ke payload lama untuk backward compatibility
    image_url = job_input.get('image_url')
    image_urls = job_input.get('image_urls', [])
    callback_url = job_input.get('callback_url')
    
    urls_to_process = []
    
    if images_input and isinstance(images_input, list):
        for img in images_input:
            if isinstance(img, dict) and img.get('url'):
                item = img.copy()
                item["url"] = img['url'].strip()
                item["type"] = img.get('type', 'chompchomp').strip()
                urls_to_process.append(item)
    else:
        # Normalisasi backward compatibility
        temp_urls = []
        if isinstance(image_urls, list):
            temp_urls.extend([url for url in image_urls if isinstance(url, str) and url.strip()])
        elif isinstance(image_urls, str) and image_urls.strip():
            temp_urls.append(image_urls.strip())
            
        if image_url and isinstance(image_url, str) and image_url.strip() and image_url not in temp_urls:
            temp_urls.insert(0, image_url.strip())
            
        for url in temp_urls:
            urls_to_process.append({
                "url": url,
                "type": "chompchomp" # Default fallback
            })
        
    if not urls_to_process:
        return {"status": "error", "message": "Input 'images' atau 'image_url' tidak ditemukan."}
        
    results_list = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for idx, item in enumerate(urls_to_process):
        url = item['url']
        model_type = item['type']
        
        # Validasi tipe model
        if model_type not in models:
            results_list.append({
                "image_url": url,
                "type": model_type,
                "status": "error",
                "message": f"Tipe model '{model_type}' tidak dikenali. Gunakan 'chompchomp' atau 'beutycharm'."
            })
            continue
            
        model = models[model_type]
        
        # Generate nama file temporary unik agar tidak bertabrakan jika diproses cepat
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_image_path = f"temp_input_{idx}_{timestamp}.jpg"
        
        try:
            # 1. Download gambar sementara dari URL
            response = requests.get(url, headers=headers, stream=True, timeout=15)
            if response.status_code != 200:
                res_item = {
                    "image_url": url,
                    "type": model_type,
                    "status": "error",
                    "message": f"Gagal mendownload gambar dari URL. Status code: {response.status_code}"
                }
                for k, v in item.items():
                    if k not in res_item:
                        res_item[k] = v
                results_list.append(res_item)
                continue
                
            with open(temp_image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            # 2. Jalankan deteksi objek dengan YOLO
            results = model(temp_image_path)
            
            # Baca gambar menggunakan OpenCV untuk anotasi
            img = cv2.imread(temp_image_path)
            if img is None:
                raise ValueError("Gagal membaca gambar menggunakan OpenCV.")
                
            # 3. Parsing hasil deteksi
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Anotasi pada gambar
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    detections.append({
                        "object": class_name,
                        "confidence": confidence,
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                    })
            
            # Urutkan dan kelompokkan deteksi
            grouped_detections = group_boxes_into_rows(detections)
            
            # Konversi gambar hasil anotasi ke base64
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            res_item = {
                "image_url": url,
                "type": model_type,
                "status": "success",
                "total_detected": len(detections),
                "detections_grouped": grouped_detections,
                "image_base64": img_base64
            }
            for k, v in item.items():
                if k not in res_item:
                    res_item[k] = v
            results_list.append(res_item)
            
        except Exception as e:
            res_item = {
                "image_url": url,
                "type": model_type,
                "status": "error",
                "message": str(e)
            }
            for k, v in item.items():
                if k not in res_item:
                    res_item[k] = v
            results_list.append(res_item)
        finally:
            # Pastikan file temporary selalu terhapus setelah selesai diproses
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except Exception as clean_e:
                    print(f"Gagal menghapus file temporary {temp_image_path}: {clean_e}")
                    
    end_time = datetime.datetime.now().timestamp()
    # Susun payload hasil akhir
    final_result = {
        "job_id": job.get('id'),
        "requests": job_input,
        "status": "success",
        "results": results_list,
        "start_time": start_time,
        "end_time": end_time,
        "exec_time": end_time - start_time
    }
    
    # 4. Kirim hasil melalui callback API jika disediakan callback_url
    if callback_url and isinstance(callback_url, str) and callback_url.strip():
        callback_sent = False
        callback_error = None
        
        # Siapkan text data (JSON metadata dimasukkan ke field 'data')
        # Hapus field 'image_base64' dari list metadata JSON agar payload tidak ganda
        metadata_results = []
        files_payload = {}
        
        for idx, res in enumerate(results_list):
            meta = res.copy()
            if "image_base64" in meta:
                # Kita hapus string base64-nya karena diganti kirim file langsung
                del meta["image_base64"] 
            metadata_results.append(meta)
            
            # Ambal data OpenCV biner gambar yang tadi sudah di-encode
            if res.get("status") == "success" and "image_base64" in res:
                _, buffer = cv2.imencode('.jpg', img) # Menggunakan instance img terkait
                # Buat nama file unik untuk field form-data key nya
                file_key = f"file_{idx}"
                filename = f"detected_{idx}_{timestamp}.jpg"
                files_payload[file_key] = (filename, buffer.tobytes(), 'image/jpeg')

        final_result["results"] = metadata_results
        
        # Coba mengirim data dengan multipart/form-data via requests
        for attempt in range(3):
            try:
                # Kirim metadata JSON sebagai string di form field 'metadata'
                # Dan kirim gambar biner di parameter 'files'
                resp = requests.post(
                    callback_url.strip(), 
                    data={"metadata": json.dumps(final_result)}, 
                    files=files_payload,
                    timeout=30
                )
                if resp.status_code in [200, 201, 202, 204]:
                    callback_sent = True
                    break
                else:
                    callback_error = f"HTTP {resp.status_code}"
            except Exception as cb_e:
                callback_error = str(cb_e)
                time.sleep(1)
        
        final_result["callback_status"] = {
            "sent": callback_sent,
            "error": callback_error
        }
        
    return final_result

# Jalankan RunPod serverless worker
runpod.serverless.start({"handler": handler})