#import flask
from flask import Flask,request,abort
import json
from app.Config import *
import requests
from app.model import model_use
#import ของ firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from random import randint
                                #ใส่ชื่อไฟล์คีย์ที่โหลดมาของ firebase
cred = credentials.Certificate("projectbot-ae15f-717708da29cb.json")
firebase_admin.initialize_app(cred)



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
        predict=model_use(message)
        # predict="การกิน" # รอ model
        # ถ้ามีโมเดลไม่ต้อง if
        # if 'นัดวัน' in message:
            # Reply_massage='เวลาที่สามารถนัดได้ : '+api
        Reply_message=Answer_Patient(predict)
        ReplyMessage(Reply_token,Reply_message,Channel_access_token)
        # else:

        ## เขียนต่อเกี่ยวกับที่จะส่งต่อ
        return request.json,200
    elif request.method=='GET':
        return 200
    


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

#ดึงข้อมูล
def Answer_Patient(predict):
    database_ref=firestore.client().document('HealthCare/'+predict)
    database_dict=database_ref.get().to_dict()
    database_list=list(database_dict.values())
    ran_answer=randint(0,len(database_list)-1)
    answer_question=database_list[ran_answer]
    answer_function=answer_question
    return answer_function


