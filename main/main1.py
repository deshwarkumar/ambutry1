import cv2
import numpy as np
import face_recognition
import os
import serial
import telepot
import io
from PIL import Image
import webbrowser
import time
from math import radians, cos, sin, asin, sqrt
from datetime import datetime  # ✅ FIXED: Needed for timestamping

# === Configuration ===
IMAGE_PATH = 'Images_Basic'
SERIAL_PORT = 'COM4'  # ⚠️ Make sure this matches your actual serial port
BAUD_RATE = 9600
TELEGRAM_TOKEN = '1741301378:AAGkYg6JUHgPIZp1m9-xaxsd6t5Y7URXhtY'
CHAT_ID = 1768829570
MAP_FILE = "map.html"
map_opened = False
SAVE_FOLDER = "saved_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

DESTINATIONS = [
    (12.9716, 77.5946),
    (17.4050, 78.4700),
    (14.6500, 77.5300),
    (16.1800, 81.1400)
]

# === Haversine Distance Calculation ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a = sin(dlat/2)**2 + cos(radians(float(lat1))) * cos(radians(float(lat2))) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# === Load known face images ===
images = []
classNames = []
mylist = os.listdir(IMAGE_PATH)
for cl in mylist:
    curImg = cv2.imread(f'{IMAGE_PATH}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
print("Loaded classes:", classNames)

# === Encode known faces ===
def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
    return encodeList

encodelistknown = find_encodings(images)
print('Encoding Complete!')

# === Serial & Telegram Bot Setup ===
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
bot = telepot.Bot(TELEGRAM_TOKEN)

# === Send frame to Telegram ===
def send_image_to_telegram(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{timestamp}.jpg"
    filepath = os.path.join(SAVE_FOLDER, filename)

    cv2.imwrite(filepath, frame)

    try:
        with open(filepath, 'rb') as image_file:
            bot.sendPhoto(CHAT_ID, photo=image_file)
    except Exception as e:
        print("Failed to send image to Telegram:", e)

# === Update map using nearest location ===
def update_google_maps_to_nearest(start_lat, start_lon):
    global map_opened
    min_distance = float('inf')
    nearest_dest = None

    for dest_lat, dest_lon in DESTINATIONS:
        dist = haversine(start_lat, start_lon, dest_lat, dest_lon)
        if dist < min_distance:
            min_distance = dist
            nearest_dest = (dest_lat, dest_lon)

    dest_lat, dest_lon = nearest_dest
    directions_url = f"https://www.google.com/maps/dir/{start_lat},{start_lon}/{dest_lat},{dest_lon}"

    html_content = f"""
    <html>
    <head><meta http-equiv="refresh" content="5"><title>Live Route</title></head>
    <body>
        <script>window.location.href = "{directions_url}";</script>
        <p>If not redirected, <a href="{directions_url}">click here</a>.</p>
    </body>
    </html>
    """

    with open(MAP_FILE, "w") as f:
        f.write(html_content)

    if not map_opened:
        webbrowser.open('file://' + os.path.realpath(MAP_FILE))
        map_opened = True

# === Main Webcam Loop ===
cap = cv2.VideoCapture(0)
cnt = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()
        else:
            name = "Unknown"

        print(f"Matched: {name}")
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        if name != "Unknown":
            ser.write(b'1\n')
            time.sleep(1)

    if ser.in_waiting:
        try:
            data = ser.readline().decode().strip()
            print("Serial Data:", data)
            status, lat, lon = data.split(',')
            lat = float(lat)
            lon = float(lon)

            if int(status) < 20:
                cnt += 1
                print('cnt:', cnt)
                if cnt > 10:
                    print('Sending image to Telegram...')
                    cnt = 0
                    send_image_to_telegram(img)
            else:
                cnt = 0

            update_google_maps_to_nearest(lat, lon)

        except Exception as e:
            print("Error parsing serial input:", e)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
