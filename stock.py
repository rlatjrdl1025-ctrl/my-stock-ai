import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime

# --- 🔍 실시간 주식 이름 자동 조회 함수 ---
def get_exact_stock_name(ticker_symbol):
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        info = ticker_data.info
        long_name = info.get('longName') or info.get('shortName') or ticker_symbol
        return long_name
    except:
        return ticker_symbol

# --- 📰 [초경량+무적] 에러 없는 글로벌 실시간 뉴스 수집 엔진 ---
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
                
                # 규칙 기반 빠른 트렌드 요약 (서버가 다운되지 않는 초경량 방식)
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

# 🌟 [즐겨찾기 완전 고정] 새로고침해도 내가 넣고 뺀 목록이 유지되도록 선언
if "favorites" not in st.session_state:
    st.session_state.favorites = ["005930.KS", "TSLA", "NVDA"] 
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("⚙️ 분석 설정")

# --- ⭐ 즐겨찾는 종목 섹션 ---
st.sidebar.subheader("⭐ 즐겨찾는 종목")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        if st.sidebar.button(f"⭐ {fav}", key=f"fav_{fav}", use_container_width=True):
            st.session_state.selected_stock = fav
else:
    st.sidebar.caption("즐겨찾기가 비어있습니다. 아래에서 추가해 보세요!")

# 종목 코드 입력창 연동
default_stock = st.session_state.get("selected_stock", "005930.KS")
my_stock = st.sidebar.text_input("1. 종목 코드 입력", value=default_stock).upper().strip()

# 🌟 즐겨찾기 추가/제거 버튼 완벽 동기화 구현
if my_stock in st.session_state.favorites:
    if st.sidebar.button("❌ 현재 종목 즐겨찾기에서 빼기", use_container_width=True):
        st.session_state.favorites.remove(my_stock)
        st.rerun()
else:
    if st.sidebar.button("➕ 현재 종목 즐겨찾기에 넣기", use_container_width=True):
        if my_stock: # 빈칸이 아닐 때만 추가
            st.session_state.favorites.append(my_stock)
            st.rerun()

months_ago = st.sidebar.slider("2. 학습 기간 설정 (개월)", min_value=3, max_value=36, value=14)

st.sidebar.markdown("---")
run_button = st.sidebar.button("종합 시장 분석 시작 🔥", use_container_width=True)

# --- 📜 최근 검색 기록 섹션 ---
st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        if st.sidebar.button(f"🕒 {hist}", key=f"hist_{hist}", use_container_width=True):
            st.session_state.selected_stock = hist
            st.rerun()

# --- 🚀 메인 분석 가동부 ---
if not run_button and "selected_stock" not in st.session_state:
    st.info(f"💡 코드 입력창에 종목을 치거나, 즐겨찾기 단추를 누른 뒤 [종합 시장 분석 시작 🔥] 버튼을 눌러주세요!")
else:
    if "selected_stock" in st.session_state and not run_button:
        my_stock = st.session_state.selected_stock
        
    if my_stock not in st.session_state.history and my_stock:
        st.session_state.history.insert(0, my_stock)
        st.session_state.history = st.session_state.history[:5]

    if "selected_stock" in st.session_state:
        del st.session_state.selected_stock

    with st.spinner("종목 정보를 실시간 조회 중..."):
        stock_display_name = get_exact_stock_name(my_stock)
    
    tab1, tab2 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스 요약"])
    
    with tab1:
        with st.spinner("AI가 데이터를 수집하고 학습하는 중입니다..."):
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
            
            data = yf.download(my_stock, start=start_date, end=end_date)
            
            if len(data) < 30:
                st.error("데이터가 부족합니다. 코드 뒤에 시장 식별자를 붙여주세요. (예: 삼성전자는 005930.KS / 카카오는 035720.KS)")
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
        st.subheader(f"📰 {stock_display_name} 관련 실시간 뉴스 핵심 요약")
        with st.spinner("시장 뉴스를 실시간으로 안전하게 파싱 중입니다..."):
            news_data = get_stock_news_light(my_stock)
            
            if not news_data:
                st.warning("현재 최신 글로벌 뉴스가 수집되지 않았습니다.")
            else:
                for news in news_data:
                    with st.container():
                        # 제목 링크 클릭 시 원문 이동 완벽 보장
                        st.markdown(f"### [{news['status']}] [{news['title']}]({news['link']})")
                        st.success(f"💬 **실시간 트렌드 요약본:** {news['summary']}")
                        st.caption(f"🔗 *제공처:* {news['publisher']}")
                        st.markdown("---")
