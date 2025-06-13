import pickle
import pyttsx3
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import ttk

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def speak_in_thread(predicted_character):
    speak(predicted_character)
    time.sleep(15)

engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)

def select_voice(x):
    global voice
    if x == 'Male':
        voice = voices[0].id
    else:
        voice = voices[1].id
    engine.setProperty('voice', voice)
    window.destroy()

window = Tk()
window.geometry("750x250")

def entry_update(text):
    entry.delete(0,END)
    entry.insert(0,text)

entry= Entry(window, width= 30, bg= "white")
entry.pack(pady=10)

button_dict={}
option= ["Female", "Male"]

button_input = None

def func(x):
    global button_input
    button_input = x
    return entry_update(x)

for i in option:
    button_dict[i] = ttk.Button(window, text=i, command=lambda x=i: select_voice(x))
    button_dict[i].pack()

window.mainloop()

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'water', 1: 'money', 2: 'I love you', 3: 'Yes', 4: 'Hello'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:  # Check if only 1 hand is detected
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]
        threading.Thread(target=speak_in_thread, args=(predicted_character,)).start()
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)