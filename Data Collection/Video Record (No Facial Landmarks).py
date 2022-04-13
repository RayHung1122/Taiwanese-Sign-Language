# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 00:29:57 2021

@author: Ray
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp
import time 



mp_holistic=mp.solutions.holistic
mp_drawimg=mp.solutions.drawing_utils
#path
DATA_PATH=r"C:/Users/Ray/Desktop/AI/Sign Language/Sign Language Dataset"
#action
#actions=np.array(['hello','thanks','iloveyou'])
#Amount of Video
no_sequences=24
#thirty frame
sequence_length=30
action=input('請輸入動作:')
def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
def draw_landmarks(image,results):
    #mp_drawimg.draw_landmarks(image,results.face_landmarks,mp_holistic.FACE_CONNECTIONS) #face
    mp_drawimg.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS) #pose
    mp_drawimg.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS) #Lhand
    mp_drawimg.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS) #Rhand
def draw_styled_landmarks(image,results):
    #mp_drawimg.draw_landmarks(image,results.face_landmarks,mp_holistic.FACE_CONNECTIONS,
    #mp_drawimg.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),#Landmark
    #mp_drawimg.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)) #Connection #face
    mp_drawimg.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
    mp_drawimg.DrawingSpec(color=(80,22,10),thickness=2,circle_radius=4),#Landmark
    mp_drawimg.DrawingSpec(color=(80,44,121),thickness=2,circle_radius=2)) #Connectioe #pose
    mp_drawimg.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
    mp_drawimg.DrawingSpec(color=(121,22,76),thickness=2,circle_radius=4),#Landmark
    mp_drawimg.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2)) #Connectione #Lhand
    mp_drawimg.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
    mp_drawimg.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=4),#Landmark
    mp_drawimg.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2)) #Connection #Rhand
def extract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()if results.pose_landmarks else np.zeros(33*4)
    #face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten()if results.face_landmarks else np.zeros(468*3)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten()if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh,rh])

for sequence in range(no_sequences):
    try:
        os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
    except:
        pass
cap=cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)as holistic:
    for sequence in range(no_sequences):
        while True:
            #reading from cam
            ret,frame=cap.read()
            #Make detection
            image,results=mediapipe_detection(frame,holistic)
            #Draw landmark
            draw_styled_landmarks(image,results)
            cv2.putText(image,f'Video Number {sequence}',(15,30),
                           cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
            cv2.putText(image,'Collecting Frame Number 0',(15,60),
                           cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
            cv2.imshow('feed',image)
            if cv2.waitKey(10) & 0xFF ==ord('x'):
                TIMER = int(2)
                prev = time.time()
                for frame_num in range(sequence_length):
                    while TIMER >= 0:
                        ret, img = cap.read()
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, str(TIMER),
                            (200, 250), font,
                            7, (255, 0, 255),
                            4, cv2.LINE_AA)
                        cv2.imshow('feed', img)
                        cv2.waitKey(1)
    
                        cur = time.time()
            
                        if cur-prev >= 1:
                            prev = cur
                            TIMER = TIMER-1  
                    else:
                        ret,frame=cap.read()
                        image,results=mediapipe_detection(frame,holistic)
                        draw_styled_landmarks(image,results)
                        cv2.putText(image,f'Video Number {sequence}',(15,30),
                               cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                        cv2.putText(image,f'Collecting Frame Number {frame_num}',(15,60),
                               cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
                        cv2.imshow('feed',image)
                        keypoints= extract_keypoints(results)
                        npy_path=os.path.join(DATA_PATH,action,str(sequence),str(frame_num))
                        np.save(npy_path,keypoints)
                        cv2.waitKey(int(1000/cap.get(cv2.CAP_PROP_FPS)))
                break


            
            
            
cap.release()
cv2.destroyAllWindows()