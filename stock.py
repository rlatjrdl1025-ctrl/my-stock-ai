import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
import urllib.parse
import requests

# --- 🌐 [보안 에러 없는 무료 한글 번역 엔진] ---
def translate_to_korean(text):
    if not text:
        return ""
    try:
        # 구글 번역 공식 무료 API 우회 파싱
        base_url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ko&dt=t&q="
        url = base_url + urllib.parse.quote(text)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            result = response.json()
            translated_text = "".join([sentence[0] for sentence in result[0] if sentence[0]])
            return translated_text
    except:
        pass
    return text # 번역 실패 시 영어 원문 그대로 반환 (안전장치)

# --- 🔍 실시간 전세계 종목 이름 자동 조회 함수 ---
def get_exact_stock_name(ticker_symbol):
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        info = ticker_data.info
        long_name = info.get('longName') or info.get('shortName') or ticker_symbol
        return long_name
    except:
        return ticker_symbol

# --- 📰 실시간 뉴스 수집 엔진 ---
def get_stock_news_light(ticker_symbol):
    news_list = []
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        yahoo_news = ticker_data.news
        if yahoo_news:
            for article in yahoo_news[:4]: # 최신 뉴스 4개 수집
                title = article.get('title') or article.get('content', {}).get('title') or '실시간 시장 속보'
                link = article.get('link') or article.get('content', {}).get('clickThroughUrl') or '#'
                pub_name = article.get('publisher') or article.get('content', {}).get('provider', {}).get('displayName') or '금융 리포트'
                summary_text = article.get('summary') or article.get('content', {}).get('summary') or title
                
                # 규칙 기반 빠른 트렌드 요약
                status = "😐 중립"
                if any(w in title.lower() or w in summary_text.lower() for w in ['up', 'growth', 'gain', 'rise', 'bull', '호재', '상승', '최고', '매수']):
                    status = "🟢 호재 (긍정)"
                elif any(w in title.lower() or w in summary_text.lower() for w in ['down', 'fall', 'loss', 'drop', 'bear', '악재', '하락', '우려', '매도']):
                    status = "🔴 악재 (부정)"
                    
                news_list.append({"title": title, "link": link, "publisher": pub_name, "status": status, "summary": summary_text})
    except:
        pass
    return news_list

# --- 🖥️ 홈페이지 레이아웃 시작 ---
st.set_page_config(page_title="나만의 주식 AI 분석기", layout="wide")
st.title("📊🕒 AI 실시간 주가 및 시장 뉴스 대시보드")

# 🌟 [즐겨찾기 저장소 완전 고정 락(Lock)] 🌟
if "favorites" not in st.session_state:
    st.session_state.favorites = ["005930.KS", "TSLA", "NVDA"] 
if "history" not in st.session_state:
    st.session_state.history = []
if "current_input" not in st.session_state:
    st.session_state.current_input = "005930.KS"

# 사이드바 설정 영역
st.sidebar.header("⚙️ 분석 설정")

# --- ⭐ 즐겨찾는 종목 섹션 ---
st.sidebar.subheader("⭐ 즐겨찾는 종목")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        # 즐겨찾기 버튼을 누르면 입력창 값이 해당 종목으로 실시간 강제 고정 및 갱신됨
        if st.sidebar.button(f"⭐ {fav}", key=f"fav_{fav}", use_container_width=True):
            st.session_state.current_input = fav
            st.rerun()
else:
    st.sidebar.caption("즐겨찾기가 비어있습니다.")

# 종목 코드 입력창 (세션과 완전 동기화)
my_stock = st.sidebar.text_input("1. 종목 코드 입력", value=st.session_state.current_input).upper().strip()
st.session_state.current_input = my_stock # 입력할 때마다 실시간 백업

# 🌟 즐겨찾기 추가/제거 버튼 완벽 실시간 동기화 🌟
if my_stock in st.session_state.favorites:
    if st.sidebar.button("❌ 현재 종목 즐겨찾기에서 빼기", use_container_width=True):
        st.session_state.favorites.remove(my_stock)
        st.rerun() # 화면을 즉시 새로고침해서 반영
else:
    if st.sidebar.button("➕ 현재 종목 즐겨찾기에 넣기", use_container_width=True):
        if my_stock: 
            st.session_state.favorites.append(my_stock)
            st.rerun() # 화면을 즉시 새로고침해서 반영

months_ago = st.sidebar.slider("2. 학습 기간 설정 (개월)", min_value=3, max_value=36, value=14)

st.sidebar.markdown("---")
run_button = st.sidebar.button("종합 시장 분석 시작 🔥", use_container_width=True)

# --- 📜 최근 검색 기록 섹션 ---
st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        if st.sidebar.button(f"🕒 {hist}", key=f"hist_{hist}", use_container_width=True):
            st.session_state.current_input = hist
            st.rerun()

# --- 🚀 메인 분석 가동부 ---
if not run_button and not st.session_state.history:
    st.info(f"💡 코드 입력창에 종목을 치거나, 즐겨찾기 단추를 누른 뒤 [종합 시장 분석 시작 🔥] 버튼을 눌러주세요!")
else:
    # 실행 버튼을 눌렀거나, 즐겨찾기/기록 클릭으로 자동 실행 유도할 때 히스토리 축적
    if my_stock and (not st.session_state.history or st.session_state.history[0] != my_stock):
        if my_stock in st.session_state.history:
            st.session_state.history.remove(my_stock)
        st.session_state.history.insert(0, my_stock)
        st.session_state.history = st.session_state.history[:5]

    with st.spinner("종목 정보를 실시간 조회 중..."):
        stock_display_name = get_exact_stock_name(my_stock)
    
    tab1, tab2 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스 요약"])
    
    with tab1:
        with st.spinner("AI가 데이터를 수집하고 학습하는 중입니다..."):
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
            
            data = yf.download(my_stock, start=start_date, end=end_date)
            
            if len(data) < 30:
                st.error("데이터가 부족합니다. 코드 뒤에 시장 식별자를 붙여주세요. (예: 삼성전자는 005930.KS / 엔비디아는 NVDA)")
            else:
                data['MA5'] = data['Close'].rolling(window=5).mean()   
                data['MA20'] = data['Close'].rolling(window=20).mean() 
                
                delta = data['Close'].diff()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / ema_down
                data['RSI'] = 100 - (100 / (1 + rs))
                
                df = data.dropna().copy()
                X = df[['Close', 'Volume', 'MA5', 'MA20', 'RSI']] 
                df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
                y = df['Target']
                
                X_today = X.iloc[[-1]] 
                X = X.iloc[:-1]
                y = y.iloc[:-1]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                
                ai_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                ai_model.fit(X_train, y_train)
                
                y_pred = ai_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🤖 AI 분석 보고서")
                    st.info(f"📈 분석 대상 종목 : **{stock_display_name} ({my_stock})**")
                    st.caption(f"📅 분석 기간 : {start_date} ~ {end_date}")
                    st.metric(label="🎯 업그레이드 AI 정확도", value=f"{accuracy * 100:.2f}%")
                    
                    tomorrow_pred = ai_model.predict(X_today)
                    current_rsi = df['RSI'].iloc[-1]
                    
                    if tomorrow_pred[0] == 1:
                        st.success(f"🔮 AI 최종 판단 : **[ 상승 예상 📈 ]** 내일 {stock_display_name} 주가는 오를 확률이 높습니다.")
                    else:
                        st.error(f"🔮 AI 최종 판단 : **[ 하락 예상 📉 ]** 내일 {stock_display_name} 주가는 떨어질 확률이 높습니다.")
                
                with col2:
                    st.subheader(f"📈 {stock_display_name} 차트 흐름")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(data['Close'], label='Price', color='blue')
                    ax.plot(data['MA5'], label='5-Day Line', color='green', linestyle=':')
                    ax.plot(data['MA20'], label='20-Day Line', color='orange', linestyle='--')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
    with tab2:
        st.subheader(f"📰 {stock_display_name} 관련 실시간 뉴스 핵심 요약 (한글 번역)")
        with st.spinner("시장 뉴스를 수집하고 실시간 한글로 번역하는 중입니다..."):
            news_data = get_stock_news_light(my_stock)
            
            if not news_data:
                st.warning("현재 최신 글로벌 뉴스가 수집되지 않았습니다.")
            else:
                for news in news_data:
                    with st.container():
                        # 🌟 실시간 한글 번역 가동
                        ko_title = translate_to_korean(news['title'])
                        ko_summary = translate_to_korean(news['summary'])
                        
                        # 제목 링크 및 요약본 한글 표출
                        st.markdown(f"### [{news['status']}] [{ko_title}]({news['link']})")
                        st.success(f"💬 **실시간 한글 요약본:** {ko_summary}")
                        st.caption(f"🔗 *제공처:* {news['publisher']} (원문 제목: {news['title']})")
                        st.markdown("---")
