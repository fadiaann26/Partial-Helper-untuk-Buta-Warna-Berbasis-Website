import numpy as np
import cv2
from flask import Flask, Response

app = Flask(__name__)

# Inisialisasi objek video capture (menggunakan kamera 0) dengan resolusi yang diinginkan
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Mengatur lebar frame menjadi 2560
cap.set(4, 720)  # Mengatur tinggi frame menjadi 1440

def generate_frames():
    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        # Mengubah frame ke format warna HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Menciptakan mask untuk deteksi warna hijau
        lower_green = np.array([35, 100, 100])  # Rentang warna hijau
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Mendefinisikan range warna merah secara lebih akurat
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Menciptakan mask untuk deteksi warna merah yang lebih akurat
        red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Melakukan operasi morfologi untuk membersihkan mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Mengubah warna merah menjadi kuning kecoklatan
        frame[np.where((red_mask == 255))] = [40, 140, 240]

        # Mengubah warna hijau menjadi krem
        frame[np.where((green_mask == 255))] = [20, 100, 200]

        # Menyimpan frame hasil ke dalam variabel 'frame'
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/deuteranopia")
def deuteranopia(): 
    # Lakukan pekerjaan Anda
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)


# Tidak terdapat sel kerucut hijau membuat pengidap kondisi ini cenderung melihat warna merah menjadi kuning kecokelatan dan warna hijau menjadi krem.
# Deuteranopia adalah kondisi ketika mata melihat warna hijau menjadi krem dan warna merah menjadi kuning kecokelatan.
