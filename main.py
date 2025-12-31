# ===============================
# THEME OCR + PRE/POST CORRECTION (FULL)
# ===============================

import cv2
import numpy as np
import easyocr
import re
import pandas as pd
import Levenshtein
from google.colab import files

# ===============================
# THEME POOL
# ===============================
THEME_POOL = [
"로봇", "휴머노이드", "의료AI", "자율주행", "감속기", "이재명", "정치", "서울시장", "대선", "정원오", "지역화폐", "전기차", "부동산", "자산", "전력", "항암","면역항암제","면역항암","RNA","mRNA","비만","비만치료제","위고비","일라이릴리",
"엔비디아", "테슬라", "스페이스x", "스페이스", "테슬라", "메타버스", "LCD", "OLED", "원격", "의료기기", "비대면", "반도체", "VR", "가상현실", "가상", "파운드리", "유전자", "진단키트", "키트", "AI", "서버", "ESS" "온디바이스", "양자",
"양저컴퓨터", "폴더블", "폴더블폰", "터치패널", "강화유리", "휴대폰", "스마트폰", "칩", "통신", "게임", "엔터", "빌보드", "안테나", "보안", "해킹", "삼성", "SK하이닉스", "SK", "지주사","증권","스테이블","스테이블코인", "IOT", "당뇨",
"치매", "알츠하이머", "치매약", "전해질", "음극재", "양극재", "음극", "양극", "방산","국방","미사일","재건","우크라이나","우크라이너재건","전쟁","우주", "우주항공", "조선", "LNG",LPG","엔진", "바이오", "제약", "줄기세포", "마이크로바이옴",
"리튬", "통신", "탄소포집", "탄소", "5G", "6G", "통신장비", "신규주", "중동재건", "남북경협", "헷지", "헷지주", "총선", "저출산", "결제", "가상화폐", "비트코인","보험","증권","은행","금융","카지노","호텔","여행","뷰티","K뷰티","화장품",
"음식", "식자재", "프랜차이즈", "웹툰", "미디어", "광고", "이커머스", "유통", "콘솔", "그래핀", "치아", "화학", "세라믹", "기계", "희토류", "구리", "알루미늄", "철강", "금","은", "금속", "미사일", "자원","해저자원","위성","로켓","드론",
"태양광", "풍력", "태양광발전", "풍력발전", "원전", "전력", "재활용"]


def h2j(text):
    CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
    JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    res = ""
    for c in text:
        if '가' <= c <= '힣':
            code = ord(c) - ord('가')
            res += CHO[code//588] + JUNG[(code//28)%21] + JONG[code%28]
        else:
            res += c
    return res

# ===============================
# THEME CORRECTOR
# ===============================
def correct_theme_from_pool(raw):
    clean = re.sub(r'[^가-힣A-Z0-9]', '', raw.upper())
    if len(clean) < 2:
        return None

    if clean in THEME_POOL:
        return clean

    cj = h2j(clean)
    best, best_sim = None, 0

    for t in THEME_POOL:
        tj = h2j(t.upper())
        dist = Levenshtein.distance(cj, tj)
        sim = 1 - dist / max(len(cj), len(tj))
        if t.startswith(clean[:1]):
            sim += 0.2
        if sim > best_sim:
            best_sim = sim
            best = t

    return best if best_sim >= 0.5 else None

# ===============================
# OCR
# ===============================
reader = easyocr.Reader(['ko','en'], gpu=True)

def extract_with_debug(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15,70,120), (45,255,255))

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w < 30 or h < 8:
            continue
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi,None,fx=3,fy=3)
        raw = "".join(reader.readtext(roi,detail=0))
        corrected = correct_theme_from_pool(raw)
        rows.append([img_path, raw, corrected])

    return rows

# ===============================
# RUN
# ===============================
uploaded = files.upload()
out = []

for fn in uploaded.keys():
    rows = extract_with_debug(fn)
    for r in rows:
        print(f"RAW ▶ {r[1]}  ==>  CORRECT ▶ {r[2]}")
        out.append(r)

df = pd.DataFrame(out, columns=["파일","보정전(OCR)","보정후(테마)"])
df.to_csv("theme_debug_result.csv", index=False, encoding="utf-8-sig")
files.download("theme_debug_result.csv")
