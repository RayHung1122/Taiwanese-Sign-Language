# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 20:46:46 2021

@author: Ray
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU,Dropout, Bidirectional,BatchNormalization,ReLU,Activation
from tensorflow.keras.callbacks import TensorBoard
from scipy import stats
from PIL import ImageFont, ImageDraw, Image
from keras import models  
import autokeras as ak

mp_holistic=mp.solutions.holistic
mp_drawimg=mp.solutions.drawing_utils
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


actions = ['星期二', '幾月幾號', '台北', '星期一', '星期三', '有', '什麼', '完了嗎', '將近', 
           '父母', '一共', '買', '家裡', '房子', '銀行', '昨天', '認識', '見她', '今天', '久',
           '比較', '一', '手語', '你', '他們', '星期五', '是嗎', '誰', '還沒有', '星期六', 
           '桃園', '朋友', '高鐵到', '吃飯', '會不會呢', '我問你', '生日', '我們兩個', '租', 
           '棒', '孩子', '零', '星期天', '明天', '運動', '星期四', '捷運站', '一樣', '上課',
           '我', '年齡', '無', '天氣', '相見', '名字']



model = Sequential()
model.add(GRU(512, return_sequences=True, activation='elu', input_shape=(60,165)))
model.add(GRU(1024, return_sequences=True, activation='elu'))
model.add(GRU(512, return_sequences=False, activation='elu'))
model.add(Dense(512, activation='elu'))
model.add(Dense(256, activation='elu'))
model.add(Dense(np.array(actions).shape[0], activation='softmax'))
model = models.load_model(r"C:/Users/Ray/Desktop/action.h5")



# 1. New detection variables
sequence=[]#collect 30 FRAME
sentence=[]
new_point=[]
threshold=0.5
predictions=[]
cap=cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5)as holistic:
    while cap.isOpened():
        #reading from cam
        ret,frame=cap.read()
        #Make detection
        image,results=mediapipe_detection(frame,holistic)
        #Draw landmark
        draw_styled_landmarks(image,results)
        keypoints=extract_keypoints(results)
        ######################
        pose=keypoints[:33*4]#pose
        hands=keypoints[33*4:]
        pose=pose.reshape(-1,4)
        hands=hands.reshape(-1,3)
        pose1=pose[:,0:3]-pose[0,0:3]
        hand1=hands[:,:]-pose[0,0:3]
        hand1[hands==0]=0
        pose1=np.concatenate([np.expand_dims(pose1[0],axis=0),pose1[11:23]],axis=0).flatten()
        all_pt=np.concatenate([pose1,hand1.flatten()],axis=0)
        new_point.append(all_pt)
        new_point=new_point[-60:]
        #new_point = np.array(new_point).reshape(126)
        
        '''
        pose=keypoints[:33*4]#pose
        hands=keypoints[33*4:]
        pose=pose.reshape(-1,4)
        hands=hands.reshape(-1,3)
        pose1=pose[:,0:3]-pose[0,0:3]
        hands=hands[:,:]-pose[0,0:3]
        pose2=np.concatenate([np.expand_dims(pose1[0],axis=0),pose1[11:23]],axis=0).flatten()
        all_pt=np.concatenate([pose2,hands.flatten()],axis=0)
        '''
        #####################
        sequence.append(keypoints)
        sequence=sequence[-60:]
        
        if len(sequence)==60:
            res=model.predict(np.expand_dims(np.array(new_point).reshape(60,165),axis=0))[0]
            predictions.append(np.argmax(res))
            #print(actions[np.argmax(res)])
            #if np.unique(predictions[-120:])[0]==np.argmax(res):
            if np.unique(predictions[-10:])[0]==np.argmax(res) and len(np.unique(predictions[-10:]))==1:
                if res[np.argmax(res)]>threshold:
                    if len(sentence)>0: #若有上個動作
                        if actions[np.argmax(res)]!=sentence[-1]: #判斷是否跟上個動作相同
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
            if len(sentence)>5:
                sentence=sentence[-5:] #抓取最後五個動作 防止串列過長
            #image=prob_viz(res,actions,image,colors)
        cv2.rectangle(image,(0,0),(640,40),(245,117,16),-1)
#         cv2.putText(image,' '.join(sentence),(3,30),
#                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        fontPath = "/Users/krama/Desktop/Sign Language/test1/TaipeiSansTCBeta-Bold.ttf"
        font = ImageFont.truetype(fontPath, 15)
        imgPil = Image.fromarray(image)
        draw = ImageDraw.Draw(imgPil)
        draw.text((3,10), ' '.join(sentence), font = font, fill = (255,255,255))
        image = np.array(imgPil)
        
        cv2.imshow('feed',image)
        if cv2.waitKey(10) & 0xFF ==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()