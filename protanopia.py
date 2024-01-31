import numpy as np
import cv2
from flask import Flask, Response

app = Flask(__name__)

# Inisialisasi objek video capture (menggunakan kamera 0) dengan resolusi yang diinginkan
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Mengatur lebar frame menjadi 2560
cap.set(4, 720)  # Mengatur tinggi frame menjadi 1440

# Inisialisasi detector wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def generate_frames():
    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        # Mengubah frame ke format warna HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mendeteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Menciptakan mask untuk deteksi warna merah
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        # Menciptakan mask untuk warna jingga dan hijau
        lower_orange = np.array([10, 100, 100])  # Rentang warna jingga
        upper_orange = np.array([60, 255, 255])
        orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        # Menciptakan mask untuk warna hijau
        lower_green = np.array([60, 100, 100])  # Rentang warna hijau
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Menciptakan mask untuk warna ungu
        lower_purple = np.array([130, 100, 100])  # Rentang warna ungu
        upper_purple = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)

        # Menciptakan mask untuk warna biru
        lower_blue = np.array([100, 100, 100])  # Rentang warna biru
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Mengubah warna merah menjadi abu-abu hitam
        frame[np.where((red_mask1 == 255))] = [0, 0, 0]
        frame[np.where((red_mask2 == 255))] = [0, 0, 0]

        # Mengubah warna jingga, hijau, dan hijau menjadi kuning
        frame[np.where((orange_mask == 255))] = [20, 100, 200]
        frame[np.where((green_mask == 255))] = [20, 100, 200]

        # Mengubah warna ungu menjadi putih
        frame[np.where((purple_mask == 255))] = [255, 255, 255]

        # Mengubah warna biru menjadi pink
        frame[np.where((blue_mask == 255))] = [255, 182, 193]

        # Menyimpan frame hasil ke dalam variabel 'frame'
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/protanopia")
def protanopia():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)
