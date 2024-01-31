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

        # Menciptakan mask untuk warna kuning
        lower_yellow = np.array([20, 100, 100])  # Rentang warna kuning
        upper_yellow = np.array([60, 255, 255])
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Menciptakan mask untuk warna biru
        lower_blue = np.array([90, 100, 100])  # Rentang warna biru
        upper_blue = np.array([120, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        # Jika mendeteksi warna biru, ubah menjadi hijau
        frame[np.where((blue_mask == 255))] = [0, 255, 0]

        # Jika mendeteksi warna kuning, ubah menjadi abu-abu keunguan
        frame[np.where((yellow_mask == 255))] = [128, 128, 128]

        # Menyimpan frame hasil ke dalam variabel 'frame'
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/tritanopia")
def tritanopia():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)
