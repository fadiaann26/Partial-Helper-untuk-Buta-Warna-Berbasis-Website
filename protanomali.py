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

        # Menciptakan mask untuk warna merah
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        # Menciptakan mask untuk warna jingga
        lower_orange = np.array([10, 100, 100])  # Rentang warna jingga
        upper_orange = np.array([60, 255, 255])
        orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        # Menciptakan mask untuk warna kuning
        lower_yellow = np.array([20, 100, 100])  # Rentang warna kuning
        upper_yellow = np.array([60, 255, 255])
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Menciptakan mask untuk warna hijau tua
        lower_dark_green = np.array([60, 100, 100])  # Rentang warna hijau tua
        upper_dark_green = np.array([90, 255, 255])
        dark_green_mask = cv2.inRange(hsv_frame, lower_dark_green, upper_dark_green)

        # Mengubah warna jingga, kuning, dan merah menjadi hijau tua
        frame[np.where((orange_mask == 255))] = [0, 128, 0]
        frame[np.where((yellow_mask == 255))] = [0, 128, 0]
        frame[np.where((red_mask1 == 255))] = [0, 128, 0]
        frame[np.where((red_mask2 == 255))] = [0, 128, 0]

        # Mengubah warna hijau menjadi hijau tidak cerah
        lower_bright_green = np.array([90, 100, 100])  # Rentang warna hijau cerah
        upper_bright_green = np.array([120, 255, 255])
        bright_green_mask = cv2.inRange(hsv_frame, lower_bright_green, upper_bright_green)
        frame[np.where((bright_green_mask == 255))] = [85, 107, 47]

        # Menyimpan frame hasil ke dalam variabel 'frame'
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/protanomali")
def protanomali():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)


#Terjadi akibat adanya gangguan fungsi fotopigmen merah sehingga warna jingga, merah, dan kuning tampak lebih gelap menyerupai warna hijau. Kondisi yang bersifat ringan ini dialami sekitar satu persen pria dan tidak begitu berpengaruh terhadap aktivitas sehari-hari.
#Penderita protanomali akan melihat warna jingga, kuning, dan merah, menjadi warna hijau. Warna hijau yang terlihat juga tidak secerah warna aslinya.
