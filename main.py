import cv2
import mediapipe as mp
import numpy as np
import math
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import io
from PIL import Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyBc5U9y1G_GBxACequbFxqa7g9z3DZmgW4")
model = genai.GenerativeModel("gemini-1.5-flash")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
socketio = SocketIO(app)

canvas = None
is_drawing = False
drawing_points = []
last_guess = ""
is_guessing = False
last_peace_time = 0

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def thumb_index_together(thumb_tip, index_tip, w, h):
    pt1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    pt2 = (int(index_tip.x * w), int(index_tip.y * h))
    return distance(pt1, pt2) < 40

def palm_open(hand_landmarks, w, h):
    tips = [4, 8, 12, 16, 20]
    base = hand_landmarks.landmark[0]
    return all(distance((landmark.x * w, landmark.y * h), (base.x * w, base.y * h)) > 80
               for landmark in [hand_landmarks.landmark[i] for i in tips])

def is_peace(landmarks, w, h):
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    thumb_out = landmarks[4].x < landmarks[2].x if landmarks[4].x < 0.5 else landmarks[4].x > landmarks[2].x
    return index_up and middle_up and ring_down and pinky_down and thumb_out

def is_thumbs_up(landmarks):
    return landmarks[4].y < landmarks[3].y < landmarks[2].y < landmarks[1].y and all(
        landmarks[i].y > landmarks[0].y for i in [8, 12, 16, 20])

def is_thumbs_down(landmarks):
    return landmarks[4].y > landmarks[3].y > landmarks[2].y > landmarks[1].y and all(
        landmarks[i].y < landmarks[0].y for i in [8, 12, 16, 20])

def smooth_line(points, alpha=0.7):
    smoothed = []
    for i in range(1, len(points)):
        smoothed_point = (
            int(points[i - 1][0] * alpha + points[i][0] * (1 - alpha)),
            int(points[i - 1][1] * alpha + points[i][1] * (1 - alpha))
        )
        smoothed.append(smoothed_point)
    return smoothed

def ask_gemini_about_image(canvas_only):
    global is_guessing
    is_guessing = True
    socketio.emit('guessing', {'is_guessing': True})
    _, buffer = cv2.imencode('.jpg', canvas_only)
    img_bytes = buffer.tobytes()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    response = model.generate_content(["What is this drawing? One word only.", img])
    is_guessing = False
    socketio.emit('guessing', {'is_guessing': False})
    return response.text.strip()

def handle_gemini_guess(canvas_img):
    socketio.emit('guessing', {'is_guessing': True})
    global last_guess
    guess = ask_gemini_about_image(canvas_img)
    last_guess = guess
    print("Gemini guess:", guess)
    socketio.emit('guess_result', {'guess': last_guess})

def generate_frames():
    global canvas, is_drawing, drawing_points, last_guess, is_guessing, last_peace_time
    cap = cv2.VideoCapture(0)
    prev_x, prev_y = -1, -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if canvas is None:
            canvas = np.zeros_like(frame)

        is_eraser = False

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            is_eraser = palm_open(hand_landmarks, w, h)

            if is_eraser:
                is_drawing = False
                if prev_x != -1 and prev_y != -1:
                    cv2.circle(canvas, (prev_x, prev_y), 30, (0, 0, 0), -1)
                prev_x, prev_y = cx, cy
            else:
                if thumb_index_together(thumb_tip, index_tip, w, h):
                    is_drawing = True
                    drawing_points.append((cx, cy))
                    if len(drawing_points) > 1:
                        smoothed = smooth_line(drawing_points)
                        for i in range(1, len(smoothed)):
                            cv2.line(canvas, smoothed[i - 1], smoothed[i], (0, 255, 0), 5)
                    prev_x, prev_y = cx, cy
                else:
                    is_drawing = False
                    drawing_points = []
                    prev_x, prev_y = -1, -1

            if is_peace(landmarks, w, h) and not is_guessing and time.time() - last_peace_time > 2:
                last_peace_time = time.time()
                socketio.start_background_task(handle_gemini_guess, canvas.copy())

            if is_thumbs_up(landmarks):
                canvas = np.zeros_like(frame)
                drawing_points.clear()
                socketio.emit('guess_feedback', {'result': 'correct', 'guess': last_guess})

            elif is_thumbs_down(landmarks):
                socketio.emit('guess_feedback', {'result': 'incorrect', 'guess': last_guess})

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            prev_x, prev_y = -1, -1

        socketio.emit('drawing_state', {'is_drawing': is_drawing, 'is_eraser': is_eraser})
        output = cv2.addWeighted(frame, 0.6, canvas, 1.0, 0)
        ret, buffer = cv2.imencode('.jpg', output)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def on_connect():
    print("Client connected!")
    socketio.emit('drawing_state', {'is_drawing': is_drawing, 'is_eraser': False})
    if last_guess:
        socketio.emit('guess_result', {'guess': last_guess})

@socketio.on('disconnect')
def on_disconnect():
    print("Client disconnected!")

if __name__ == '__main__':
    socketio.run(app, port=8000, debug=True)