import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image

import pyttsx3
import datetime
import speech_recognition as sr
import os
import time
import re
import random
import joblib
from joblib import dump, load

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier



engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
# print(voices[1].id)

def speak(audio):
	engine.say(audio)
	engine.runAndWait()


def name():
	speak("Hello Sir")


def takeCommand():
	# It take input and return output
	global emotions
	r = sr.Recognizer()
	with sr.Microphone() as source:
		print("Listening...")
		r.pause_threshold = 1
		r.energy_threshold = 200
		audio = r.listen(source)

	try:
		print("recognizing...")
		query = r.recognize_google(audio, language='en-us')
		print(f"User said: {query}\n")

	except Exception as e:
		print(e)
		print("Say Again")
		return "None"
	return query

if __name__== "__main__":
	name()
	while True:
		query = takeCommand().lower()
		if "music on my mood" in query:
			model = tf.keras.models.load_model("facecnn.model")

			face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			cap = cv2.VideoCapture(0)
			img_counter = 0
			while True:
			    ret, test_img = cap.read()
			    if not ret:
			        continue
			    
			    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
			    
			    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
			    
			    for (x,y,w,h) in faces:
			        img = cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),2)
			        roi_gray = gray_img[y:y+h, x:x+w]
			        roi_gray = cv2.resize(roi_gray, (48, 48))
			        img_pixels = image.img_to_array(roi_gray)
			        img_pixels = np.expand_dims(img_pixels, axis=0)
			        img_pixels /= 255.0
			        
			        predictions = model.predict(img_pixels)
			        max_index = np.argmax(predictions[0])
			        emotions = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
			        predict_emo = emotions[max_index]
			        # print(predict_emo)
			        cv2.putText(test_img, predict_emo, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
			    resized_img = cv2.resize(test_img, (1000, 700))
			    cv2.imshow("Emotion Detect", resized_img)

			    if cv2.waitKey(33) == ord('a'):
			        # img_name = "predict_img{}.png".format(img_counter)
			        # cv2.imwrite(img_name, test_img)
			        # print("{} written!".format(img_name))
			        # img_counter += 1
			        break

			cap.release()
			cv2.destroyAllWindows()


			if predict_emo == "happy":
				speak(f"You are {predict_emo}")
				music_dir = "E:\\face\\music\\song\\happy"
				songs = os.listdir(music_dir)
				os.startfile(os.path.join(music_dir, random.choice(songs)))
			elif predict_emo == "neutral":
				speak(f"You are {predict_emo}")
				music_dir = "E:\\face\\music\\song\\neutral"
				songs = os.listdir(music_dir)
				os.startfile(os.path.join(music_dir, random.choice(songs)))
		

		elif "classify news" in query:
			model = load("model.joblib")
			label = ['business', 'tech', 'politics', 'sport', 'entertainment']
			pred = model.predict([input()])
			news = label[(pred[0])]
			speak(f"{news} news")

		elif "wake up" in query:
			name()
		
