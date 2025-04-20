# ✋ DrawDrop - Gesture-Controlled AI Sketch Game

**DrawDrop** is a real-time, gesture-controlled drawing and guessing game powered by **MediaPipe**, **OpenCV**, **Flask-SocketIO**, and **Google Gemini**. Draw using your **hands in the air** — no touchscreen, no mouse — and let AI guess your sketches!


## 🖥️ Live Preview

![demo](demo.gif)

---

## 🚀 Features

- 👆 **Pinch to Draw** – Touch thumb and index finger together to draw
- ✋ **Palm to Erase** – Open your hand to switch to eraser mode
- ✌️ **Peace Gesture** – Triggers Google Gemini to guess your drawing

---

## 🧠 Powered By

- **Google Gemini** (via `google.generativeai`) for interpreting your drawing and guessing what it is
- **MediaPipe Hands** for gesture recognition
- **Flask + Flask-SocketIO** for real-time server-client interaction
- **OpenCV** for frame processing and drawing
