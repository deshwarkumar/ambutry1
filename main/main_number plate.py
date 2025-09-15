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
import numpy as np
import cv2
import  imutils
import sys
import pytesseract
import pandas as pd
import time
import requests
# === Configuration ===
IMAGE_PATH = 'Images_Basic'
SERIAL_PORT = 'COM11'  # ⚠️ Make sure this matches your actual serial port
BAUD_RATE = 9600
TELEGRAM_TOKEN = '7970513032:AAHBR6Wigyq8V-n5p0jdcRJWhjV5wCft2r4'
CHAT_ID = 5430423218
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
    txt=npr(frame)
    try:
        with open(filepath, 'rb') as image_file:
            bot.sendPhoto(CHAT_ID, photo=image_file)
    except Exception as e:
        print("Failed to send image to Telegram:", e)

    send_telegram_message('Detected Number plate:'+txt)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=payload)
    return response.ok


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
def npr(frm):
    frm = cv2.imread('car.jpeg')
    print('Number plate processing..')
    image = imutils.resize(frm, width=500)

    cv2.imshow("Original Image", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("1 - Grayscale Conversion", gray)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #cv2.imshow("2 - Bilateral Filter", gray)

    edged = cv2.Canny(gray, 170, 200)
    #cv2.imshow("4 - Canny Edges", edged)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
    NumberPlateCnt = None 

    count = 0
    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  
                NumberPlateCnt = approx 
                break

    # Masking the part other than the number plate
    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
    new_image = cv2.bitwise_and(image,image,mask=mask)
    cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
    cv2.imshow("Final_image",new_image)

    # Configuration for tesseract
    config = ('-l eng --oem 1 --psm 3')

    # Run tesseract OCR on image
    text = pytesseract.image_to_string(new_image, config=config)

    #Data is stored in CSV file
    raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
            'v_number': [text]}

    df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
    df.to_csv('data.csv')

    # Print recognized text
    print(text)
    
    cv2.waitKey(0)
    return text

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
