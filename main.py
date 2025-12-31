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
    # (네가 준 THEME_POOL 그대로 — 생략 없음)
    "로봇","휴머노이드","의료로봇","물류로봇",
    "로봇부품","로봇감속기","공장자동화","스마트팩토리",
    "이재명","정치테마","대선","지역화폐","주택","부동산",
    "전기차","차량용반도체","자율주행",
    "전력반도체","전력",  # ✅ 누락 콤마 오류 수정
    "항암","비만치료제","mRNA","백신","RNA치료","유전자치료","세포치료","CAR-T",
    "의료기기","진단키트",
    "헬스케어","의료AI","원격",
    "디스플레이","LCD","OLED","마이크로LED","플렉서블디스플레이",
    "VR","메타버스",
    "반도체","시스템반도체","메모리반도체","비메모리","파운드리",

    "AI","온디바이스AI","AI반도체","AI서버",
    "데이터센터","빅데이터","클라우드","엣지컴퓨팅","양자컴퓨팅",
    "사이버보안","정보보안","블록체인","IoT","스마트시티","디지털트윈",
    "스마트폰","스마트폰부품","모바일부품",
    "스마트폰카메라","카메라모듈","이미지센서","모바일디스플레이",
    "폴더블폰","힌지","터치패널","강화유리",
    "모바일메모리","OLED","스페이스",  # ✅ 문자열 붙음/구문 오류 수정
    "통신칩","안테나",
    "스마트폰배터리","충전기","모바일OS","안드로이드",
    "모바일게임","모바일콘텐츠","스페이스",  # ✅ 누락 콤마 오류 수정
    "통신장비","5G","6G","위성통신",
    "2차전지","전고체배터리","리튬이온배터리","LFP","NCM","리튬","니켈","코발트","망간",
    "음극재","양극재","전해질","분리막","배터리장비","배터리재활용","ESS",
    "수소","수소연료전지","태양광","풍력","해상풍력","원전","SMR",
    "전력설비","스마트그리드","탄소중립","탄소포집","탄소배출권",
    "바이오","제약","바이오시밀러","마이크로바이옴","재생의료","줄기세포",
    "당뇨","치매","희귀질환","신약개발",
    "자동차","자동차부품","라이다","레이더","전기선박","드론",
    "우주","우주항공","항공우주","민간우주","우주산업","우주개발",
    "위성","위성통신","위성부품","위성영상","소형위성","정찰위성",
    "발사체","재사용로켓","항공","항공부품","항공기엔진",
    "조선","조선업","조선기자재","LNG선","LPG선","컨테이너선","유조선","VLCC",
    "친환경선박","암모니아선","수소선박","이중연료엔진","선박엔진",
    "해양플랜트","해저케이블","해저자원",
    "방산","국방","미사일",
    "건설","건설기계","재건","철강","비철금속","알루미늄","구리","희토류",
    "기계","공작기계","플랜트","산업가스","화학","정밀화학",
    "고순도소재","세라믹","나노소재","그래핀","탄소섬유","복합소재",
    "게임","게임개발","게임퍼블리싱","PC게임","콘솔게임",
    "캐주얼게임","P2E","블록체인게임","게임엔진",
    "유통","이커머스","플랫폼","콘텐츠","엔터테인먼트","웹툰","미디어","광고",
    "푸드테크","프랜차이즈","화장품","K-뷰티","여행","호텔","카지노",
    "금융지주","은행","증권","보험","핀테크","가상자산","결제",
    "총선","저출산","고령화","남북경협","우크라이나재건","중동재건",
    "원자재","곡물","농업","스마트팜","식량안보","기후변화",
    "신규주","IPO"
]

# ===============================
# JAMO
# ===============================
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
