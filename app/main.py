#import flask
from flask import Flask,request,abort
import json
from app.Config import *
import requests
# from app.model import model_use
#import ของ firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from random import randint

#โมเดล
import pandas as pd
import sklearn
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
# from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from gensim.models import KeyedVectors
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
                                #ใส่ชื่อไฟล์คีย์ที่โหลดมาของ firebase
cred = credentials.Certificate("projectbot-ae15f-717708da29cb.json")
firebase_admin.initialize_app(cred)

word2vec_model = KeyedVectors.load_word2vec_format('LTW2V_v0.1.bin',binary=True,unicode_errors='ignore')

app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])

def webhook():

    if request.method=='POST':
        print(request.json)
        api="aa.bb"
        payload = request.json
        Reply_token = payload['events'][0]['replyToken']
        print(Reply_token)
        message=payload['events'][0]['message']['text']
        print(message)
        text=message
        if 'การเตรียมตัวก่อนและหลังผ่าตัดเต้านม' in message:
            Reply_message="การปฏิบัติตนก่อนผ่าตัดมีวิธีดังนี้ 1.ทำความสะอาดร่างกายของโดยการอาบน้ำ สระผม ถ้าทาเล็บไว้ให้ ล้างสีเล็บออก 2.งดอาหารและน้ำดื่มหลังเที่ยงคืนก่อนวันผ่าตัดเพื่อให้กระเพาะอาหารว่าง และป้องกันการสำลักเศษอาหารและน้ำออกมาระหว่างผ่าตัด และหลังผ่าตัด 3.ฝึกการหายใจเข้าทางจมูกลึกๆ และหายใจออกทางปากช้าๆ เพื่อให้ปอด ขยายตัวได้ดี 4.ฝึกการไอที่ถูกวิธี โดยการสูดลมหายใจเข้าลึกๆ กลั้นหายใจเล็กน้อย (2-3 วินาที) แล้วไอออกมา การใช้วิธีนี้จะช่วยให้เสมหะขับออกได้ดี"
        elif 'การรักษามะเร็งเต้านม' in message:
            Reply_message="วิธีการผ่าตัดเต้านม มีหลายแบบ ได้แก่ 1.วิธีการผ่าตัดเต้านมออกทั้งเต้า เป็นการผ่าตัดเลาะเนื้อเต้านมทั้งหมดออก โดยไม่ได้เลาะต่อมน้ำเหลือง บริเวณรักแร้ 2.วิธีการผ่าตัดเต้านมออกทั้งเต้าและเลาะต่อมน้ำเหลืองรักแร้ เป็นการผ่าตัดเลาะเนื้อเต้านมทั้งหมดออก 3.วิธีการผ่าตัดเต้านมแบบสงวนหรือเก็บเต้านมร่วมกับการฉายรังสี เป็นการผ่าตัดก้อนมะเร็งและเนื้อเยื่อปกติรอบก้อนมะเร็งนั้นออก 4.การทำศัลยกรรมสรางเต้านมใหม่ เพื่อแทนเต้านมที่ตัดออกให้เหมือนกับข้างที่เหลือ เพื่อลดผลกระทบทาง ภาพลักษณ์และเพิ่มคุณภาพชีวิตผู้ป่วย"
        elif 'อาการเบื้องต้นของมะเร็งเต้านม' in message:
            Reply_message="มะเร็งเต้านมระยะเริ่มต้นไม่มีอาการเจ็บ เเต่เเนะนำให้พบแพทย์ถ้ามีอาการเจ็บเต้านม โดยเฉพาะคลำก้อนได้"
        elif 'ควรจะเริ่มทำแมมโมแกรมเมื่อไหร่' in message:
            Reply_message="แนะนำว่าให้เริ่มทำเมื่ออายุ 35 ปี และทำอีกทุก 2-3 ปี จนเมื่ออายุ 40 ปี แล้วให้ทำทุกปี และอายุ 50 ปีขึ้นไปให้ทำทุก 1-2 ปี เพราะจากสถิติผู้ป่วยมะเร็งเต้านมเริ่มพบมากตั้งเเต่อายุ 35 ปี"
        else:

            df=pd.read_csv('data-project2.csv')
            df.columns=['input_text','labels']
            data_df=df
            #เปลี่ยนตัวอักษรตัวใหญ่เป็นตัวเล็ก
            data_df['cleaned_labels']=data_df['labels'].str.lower()

            #เอา labels ออก
            data_df.drop('labels',axis=1,inplace=True)

            data_df=data_df[data_df['cleaned_labels']!='garbage']


            cleaned_input_text=data_df['input_text'].str.strip()
            cleaned_input_text=cleaned_input_text.str.lower()

            data_df['cleaned_input_text']=cleaned_input_text
            data_df.drop('input_text',axis=1,inplace=True)

            data_df=data_df.drop_duplicates("cleaned_input_text",keep="first")

            input_text=data_df["cleaned_input_text"].tolist()
            labels=data_df["cleaned_labels"].tolist()
            train_text,test_text,train_labels,test_labels=train_test_split(input_text,labels,train_size=0.8,random_state=42)
            loaded_model = tf.keras.models.load_model('my_model')
            # tokenize
            word_seq = word_tokenize(text)
            # map index
            word_indices = map_word_index(word_seq)
            # padded to max_leng
            padded_wordindices = pad_sequences([word_indices], maxlen=12, value=0)
            # predict to get logit
            logit = loaded_model.predict(padded_wordindices, batch_size=32)
            unique_labels = set(train_labels)
            index_to_label = [label for label in sorted(unique_labels)]
            # get prediction
            predict = [ index_to_label[pred] for pred in np.argmax(logit, axis=1) ][0]
        
            Reply_message=Answer_Patient(predict)
        # predict=model_use(message)
        # predict="การกิน" # รอ model
        # ถ้ามีโมเดลไม่ต้อง if
        # if 'นัดวัน' in message:
            # Reply_massage='เวลาที่สามารถนัดได้ : '+api
        ReplyMessage(Reply_token,Reply_message,Channel_access_token)
        # else:

        ## เขียนต่อเกี่ยวกับที่จะส่งต่อ
        return request.json,200
    elif request.method=='GET':
        return "this is method GET!!",200
    else:
        abort(400)
    


def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'
    
    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200

def map_word_index(word_seq):
 
  indices = []
  for word in word_seq:
    if word in word2vec_model.vocab:
      indices.append(word2vec_model.vocab[word].index + 1)
    else:
      indices.append(1)
  return indices

#ดึงข้อมูล
def Answer_Patient(predict):
    database_ref=firestore.client().document('HealthCare/'+predict)
    database_dict=database_ref.get().to_dict()
    database_list=list(database_dict.values())
    ran_answer=randint(0,len(database_list)-1)
    answer_question=database_list[ran_answer]
    answer_function=answer_question
    # print(answer_function)
    return answer_function


