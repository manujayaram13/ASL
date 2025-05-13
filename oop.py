import cv2
import mediapipe as mp
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define gesture conditions
def folded(tip, pip):
    return tip.y > pip.y

def extended(tip, pip):
    return tip.y < pip.y

def recognize_gesture(landmarks):
    # Extract hand landmarks
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    
    # A: Thumb folded, other fingers extended
    if all(folded(landmarks[i], landmarks[i - 2]) for i in [8, 12, 16, 20]) and thumb_tip.x < thumb_ip.x:
        return "A"
    
    # B: Fist (all fingers folded)
    if all(folded(landmarks[i], landmarks[i - 2]) for i in [8, 12, 16, 20]):
        return "B"
    
    # C: Thumb and fingers curved
    if (abs(thumb_tip.x - index_tip.x) < 0.05 and
        abs(index_tip.x - middle_tip.x) < 0.05 and
        abs(middle_tip.x - ring_tip.x) < 0.05):
        return "C"

    # D: Thumb folded, others extended
    if (folded(thumb_tip, thumb_ip) and
        extended(index_tip, index_pip) and
        extended(middle_tip, middle_pip) and
        extended(ring_tip, ring_pip) and
        extended(pinky_tip, pinky_pip)):
        return "D"

    # E: All fingers curled, fist-like shape
    if all(folded(landmarks[i], landmarks[i - 2]) for i in [8, 12, 16]) and extended(thumb_tip, thumb_ip):
        return "E"
    
    # F: Thumb & index touching, others extended
    if (abs(thumb_tip.x - index_tip.x) < 0.05 and
        extended(middle_tip, middle_pip) and
        extended(ring_tip, ring_pip) and
        extended(pinky_tip, pinky_pip)):
        return "F"

    # G: Index extended horizontally, rest folded
    if (extended(index_tip, index_pip) and
        all(folded(landmarks[i], landmarks[i - 2]) for i in [12, 16, 20])):
        return "G"

    # H: Index & middle extended, rest folded
    if (extended(index_tip, index_pip) and extended(middle_tip, middle_pip) and
        folded(ring_tip, ring_pip) and folded(pinky_tip, pinky_pip)):
        return "H"

    # I: Pinky extended only
    if (extended(pinky_tip, pinky_pip) and
        all(folded(landmarks[i], landmarks[i - 2]) for i in [8, 12, 16])):
        return "I"

    # J: Like I + motion based, skip for now
    # return "J"

    # K: Index & middle up, thumb pointing at middle
    if (extended(index_tip, index_pip) and extended(middle_tip, middle_pip) and
        folded(ring_tip, ring_pip) and folded(pinky_tip, pinky_pip)):
        return "K"

    # L: Thumb and index form L, others folded
    if (extended(index_tip, index_pip) and extended(thumb_tip, thumb_ip) and
        all(folded(landmarks[i], landmarks[i - 2]) for i in [12, 16, 20])):
        return "L"

    # M: Thumb under 3 fingers
    if (folded(index_tip, index_pip) and folded(middle_tip, middle_pip) and folded(ring_tip, ring_pip) and
        pinky_tip.y < pinky_pip.y):
        return "M"

    # N: Thumb under 2 fingers
    if (folded(index_tip, index_pip) and folded(middle_tip, middle_pip) and
        extended(ring_tip, ring_pip) and extended(pinky_tip, pinky_pip)):
        return "N"

    # O: All fingertips close to form a circle
    if (abs(thumb_tip.x - index_tip.x) < 0.05 and
        abs(index_tip.x - middle_tip.x) < 0.05 and
        abs(middle_tip.x - ring_tip.x) < 0.05):
        return "O"

    # P: Similar to K but palm down (you can add orientation logic)
    # return "P"

    # Q: Like G but palm down (you can add orientation logic)
    # return "Q"

    # R: Index and middle crossed, others folded
    # Hard to detect with just position, needs depth check

    # S: Fist (all fingers folded)
    if all(folded(landmarks[i], landmarks[i - 2]) for i in [8, 12, 16, 20]):
        return "S"

    # T: Thumb between index and middle (hard but approx)
    if (folded(index_tip, index_pip) and folded(middle_tip, middle_pip) and
        thumb_tip.y > index_pip.y):
        return "T"

    # U: Index and middle together & up
    if (extended(index_tip, index_pip) and extended(middle_tip, middle_pip) and
        abs(index_tip.x - middle_tip.x) < 0.03):
        return "U"

    # V: Index and middle apart
    if (extended(index_tip, index_pip) and extended(middle_tip, middle_pip) and
        abs(index_tip.x - middle_tip.x) > 0.06):
        return "V"

    # W: Index, middle, ring up
    if all(extended(landmarks[i], landmarks[i - 2]) for i in [8, 12, 16]) and folded(pinky_tip, pinky_pip):
        return "W"

    # X: Index bent, rest folded
    if (index_tip.y > index_pip.y and
        all(folded(landmarks[i], landmarks[i - 2]) for i in [12, 16, 20])):
        return "X"

    # Y: Thumb and pinky out
    if (extended(thumb_tip, thumb_ip) and extended(pinky_tip, pinky_pip) and
        all(folded(landmarks[i], landmarks[i - 2]) for i in [8, 12, 16])):
        return "Y"

    # Z: Skip for now (requires motion tracking)
    
    return "Unknown"

# Initialize the webcam
cap = cv2.VideoCapture(0)

def detect_and_classify(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        landmarks = result.multi_hand_landmarks[0].landmark
        gesture = recognize_gesture(landmarks)

        # Show recognized gesture text
        cv2.putText(frame, f"Detected: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# UI Setup with Tkinter
def start_camera():
    def update():
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame = detect_and_classify(frame)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)
        lbl.after(10, update)

    update()

# Tkinter Window
root = Tk()
root.title("Sign Language Translator")
lbl = Label(root)
lbl.pack()
Button(root, text="Start", command=start_camera).pack()

root.mainloop()
