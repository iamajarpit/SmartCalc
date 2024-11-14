from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os

# Load the environment variables from .env
load_dotenv()

# Initialize gemini with the API key from the .env file
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the HandDetector class with the updated parameters
detector = HandDetector(maxHands=1, detectionCon=0.75, minTrackCon=0.75)

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

def initialize_canvas(frame):
    return np.zeros_like(frame)

def process_hand(hand):
    lmList = hand["lmList"]
    bbox = hand["bbox"]
    center = hand['center']
    handType = hand["type"]
    fingers = detector.fingersUp(hand)
    return lmList, bbox, center, handType, fingers

def weighted_average(current, previous, alpha=0.5):
    return alpha * current + (1 - alpha) * previous

response_text = None

def send_to_ai(model, canvas, fingers):
    global response_text
    if fingers[4] == 1:
        image = Image.fromarray(canvas)
        response = model.generate_content(["solve this math problem", image])
        response_text = response.text if response else None

# Initialize variables
prev_pos = None
drawing = False
points = []
smooth_points = None

# Initialize canvas
_, frame = cap.read()
canvas = initialize_canvas(frame)

def video_stream():
    global prev_pos, drawing, points, smooth_points, canvas

    while True:
        success, img = cap.read()

        if not success:
            print("Failed to capture image")
            break

        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img, draw=True, flipType=True)

        if hands:
            hand = hands[0]
            lmList, bbox, center, handType, fingers = process_hand(hand)

            index_tip = lmList[8]
            thumb_tip = lmList[4]

            if fingers[1] == 1 and fingers[2] == 0:
                current_pos = np.array([index_tip[0], index_tip[1]])
                if smooth_points is None:
                    smooth_points = current_pos
                else:
                    smooth_points = weighted_average(current_pos, smooth_points)
                smoothed_pos = tuple(smooth_points.astype(int))

                if drawing:
                    points.append(smoothed_pos)
                prev_pos = smoothed_pos
                drawing = True
            elif fingers[1] == 1 and fingers[2] == 1:
                drawing = False
                prev_pos = None
                points = []
                smooth_points = None
            elif fingers[0] == 1:
                canvas = initialize_canvas(img)
                points = []
                drawing = False
                prev_pos = None
                smooth_points = None
            elif fingers[4] == 1:
                send_to_ai(model, canvas, fingers)

        if len(points) > 1 and drawing:
            cv2.polylines(canvas, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=5)

        img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_response(request):
    global response_text
    return JsonResponse({'response': response_text})
