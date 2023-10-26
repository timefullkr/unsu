from bs4 import BeautifulSoup
from urllib.request import urlopen
import os
import requests
import random
import string
import datetime
import re

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tiktoken_len(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def image_url_to_save(image_url, folder_name="", image_name=""):
   # 이미지 저장  
    if image_url    =="" :  return False
    if folder_name  =="":  folder_name="." 
    if image_name   =="":   image_name ="image-" + rnd_str(5) +".jpg"
    if not os.path.exists(folder_name):   os.makedirs(folder_name)
 
    response = requests.get(image_url)

    file_path = os.path.join(folder_name, image_name)

    file=open(file_path, "wb")

    file.write(response.content)
    return True


def rnd_str(n=5):
    return ''.join(random.choices(string.digits, k=n))
def getMealMenu(today="",period=""):


    if today=="":
       today = str(datetime.date.today())
    period="lunch" if  period =="" else  period
    meal_name=[ "breakfast","lunch","dinner"]
    t=today.split("-")
    url=f"http://jeju-s.jje.hs.kr/jeju-s/food/{t[0]}/{t[1]}/{t[2]}/{period}"

    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")
    bap = soup.select(".ulType_food > li:nth-child(2) > dl > dd")[0]
    image_url=""
    if  soup.select(".food_img> img"): 
        image=soup.select(".food_img> img")[0].get("src") 
        image_url="http://jeju-s.jje.hs.kr" + image
    menu=str(bap).split("<br/>")
    tmp=[]
 
    for m in menu:
        m=re.sub("<[^<>]*>", "",m)
        if m !='' and not "축산물" in m :
       
          tmp.append( m )

    
    return {"item":tmp,"image":image_url} 


def school_schedule(year):

    ret =f"다음은 제주과학고등학교 {year}학년도  학교 행사 또는 학사 일정이다. 학사 일정은  '행사날자: 행사명'로 구성되어 있다. 학교행사 또는 학사일정 등에 질문울 받으면 다음 데이터를 참고하여 대답한다. \n학사 일정 시작"
    section=[1,2] 
    for hakgi in section:
        html = urlopen(f"https://jeju-s.jje.hs.kr/jeju-s/0202/schedule?section={hakgi}")
        soup = BeautifulSoup(html, "html.parser")
        bap = soup.select("a")
       
        for b in bap:
            #"""<a href="#insSchl" onclick="javascript:setData('230148', '2023/03/02','2023/03/02','개교기념일','null')">02일&nbsp;:&nbsp;개교기념일</a>"""

            if  ':' in b.text:
                #javascript:setData('244463', '2023/09/06','2023/09/06','전국연합학력평가(1/2학년)','O')
                bb=b.get("onclick")
                t=bb.replace("'","")
                t=t.split(",")
                t.pop(0)
                t.pop()
                t = [e.strip() for e in t]


                
               
                    
                if t[0] != t[1] :
                    tt=t[0] + "~" + t[1] + ":" + t[2] 
                else:
                    tt=t[0] + ":" + t[2]     
                ret += tt + ", "   
    return ret.replace("&","-").replace("\n","-")  + "학사일정 종료."
if __name__ == "__main__":
    ret=school_schedule()
    print(ret)
    print("="*100)
    ret=getMealMenu("","")
    print(ret)
