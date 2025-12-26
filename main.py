from ultralytics import YOLO
import cv2
from pyzbar.pyzbar import decode
import numpy as np
import webbrowser
from datetime import datetime
import os

model = YOLO("best_final.pt")

def open_qr_content(data):
    if data.startswith(('http://', 'https://', 'www.')):
        if not data.startswith('http'):
            data = 'https://' + data
        webbrowser.open(data)
    elif data.startswith('mailto:') or ('@' in data and not data.startswith('mailto:')):
        data = 'mailto:' + data if not data.startswith('mailto:') else data
        webbrowser.open(data)
    elif data.startswith('geo:'):
        coords = data.replace('geo:', '').split(',')
        if len(coords) >= 2:
            webbrowser.open(f"https://www.google.com/maps?q={coords[0]},{coords[1]}")
    else:
        print(f"Nội dung QR: {data}")

def scan_with_yolo(frame):
    results = model(frame, conf=0.3, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        dec = decode(crop)
        if dec:
            return dec[0].data.decode('utf-8'), (x1, y1, x2, y2)
    return None, None

def scan_with_pyzbar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dec = decode(gray)
    if dec:
        pts = dec[0].polygon
        if len(pts) == 4:
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            return dec[0].data.decode('utf-8'), pts
    return None, None

def process_image(image_path):
    if not os.path.exists(image_path):
        print("Không tìm thấy file ảnh!")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Không thể đọc ảnh!")
        return

    display = frame.copy()

    # Quét bằng YOLO
    text_yolo, box_yolo = scan_with_yolo(frame)
    if text_yolo:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] YOLO phát hiện")
        open_qr_content(text_yolo)
        if box_yolo:
            x1, y1, x2, y2 = box_yolo
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 8)
        cv2.putText(display, f"YOLO: {text_yolo}", (50, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1.8, (0, 255, 0), 5)

    # Quét bằng pyzbar
    text_zbar, pts_zbar = scan_with_pyzbar(frame)
    if text_zbar:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] pyzbar phát hiện")
        if text_yolo != text_zbar:
            open_qr_content(text_zbar)
        if pts_zbar is not None:
            cv2.polylines(display, [pts_zbar], True, (0, 0, 255), 6)
        cv2.putText(display, f"pyzbar: {text_zbar}", (50, 180),
                    cv2.FONT_HERSHEY_COMPLEX, 1.6, (0, 0, 255), 4)

    # Resize nếu ảnh quá lớn để hiển thị thoải mái
    height, width = display.shape[:2]
    if width > 1200:
        scale = 1200 / width
        new_size = (int(width * scale), int(height * scale))
        display = cv2.resize(display, new_size)

    cv2.imshow(f"Ket qua quet QR - {os.path.basename(image_path)}", display)
    print("Nhấn Q để đóng cửa sổ ảnh.")
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def live_camera_scan():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    print("Đang quét thời gian thực")
    print("Bấm phím 'Q' hoặc 'ESC' để thoát")

    # Tạo cửa sổ trước
    cv2.namedWindow("YOLO (xanh) vs pyzbar (do) - Bam Q de thoat", cv2.WINDOW_NORMAL)
    # Đặt chế độ toàn màn hình
    cv2.setWindowProperty("YOLO (xanh) vs pyzbar (do) - Bam Q de thoat", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_qr = None
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        current_time = datetime.now().timestamp()

        # YOLO
        text_yolo, box_yolo = scan_with_yolo(frame)
        if text_yolo:
            if text_yolo != last_qr or current_time - last_time > 2:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] YOLO")
                open_qr_content(text_yolo)
                last_qr = text_yolo
                last_time = current_time
            if box_yolo:
                x1, y1, x2, y2 = box_yolo
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 8)
            cv2.putText(display, f"YOLO", (50, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1.8, (0, 255, 0), 5)

        # pyzbar
        text_zbar, pts_zbar = scan_with_pyzbar(frame)
        if text_zbar and text_yolo != text_zbar:
            if text_zbar != last_qr or current_time - last_time > 2:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] pyzbar:")
                open_qr_content(text_zbar)
                last_qr = text_zbar
                last_time = current_time
            if pts_zbar is not None:
                cv2.polylines(display, [pts_zbar], True, (0, 0, 255), 6)
            cv2.putText(display, f"pyzbar", (50, 180),
                        cv2.FONT_HERSHEY_COMPLEX, 1.6, (0, 0, 255), 4)

        cv2.imshow("YOLO (xanh) vs pyzbar (do) - Bam Q de thoat", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q hoặc ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# main
if __name__ == "__main__":
    print("1. Quét thời gian thực từ camera ")
    print("2. Chọn ảnh từ máy để quét QR")
    print("0. Thoát chương trình")
    print("-" * 50)

    while True:
        choice = input("Chọn chức năng (0-2): ").strip()

        if choice == "1":
            live_camera_scan()
        elif choice == "2":
            print("\nDán đường dẫn đầy đủ đến file ảnh ")
            path = input("Nhập đường dẫn ảnh: ").strip().strip('"\'')
            if path:
                process_image(path)
        elif choice == "0":
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng chọn lại.")
        
        print("\n" + "-" * 50)