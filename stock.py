import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime

# --- 🔍 [전 세계 모든 종목 연동] 실시간 주식 이름 자동 조회 함수 ---
def get_exact_stock_name(ticker_symbol):
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        # 야후 파이낸스 프로필에서 기업의 진짜 공식 풀네임을 긁어옵니다.
        info = ticker_data.info
        long_name = info.get('longName') or info.get('shortName') or ticker_symbol
        return long_name
    except:
        return ticker_symbol

# --- 🤖 구글 공식 글로벌 AI 요약 모델 ---
@st.cache_resource
def load_summary_model():
    from transformers import pipeline
    return pipeline("summarization", model="google/pegasus-xsum")

def get_stock_news_ai(ticker_symbol):
    news_list = []
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        yahoo_news = ticker_data.news
        if yahoo_news:
            for article in yahoo_news[:3]:
                title = article.get('title') or article.get('content', {}).get('title') or '실시간 시장 속보'
                link = article.get('link') or article.get('content', {}).get('clickThroughUrl') or '#'
                pub_name = article.get('publisher') or article.get('content', {}).get('provider', {}).get('displayName') or '금융 리포트'
                summary_text = article.get('summary') or article.get('content', {}).get('summary') or title
                desc = f"[{pub_name} 발간] {summary_text}"
                
                status = "😐 중립"
                if any(w in title.lower() or w in desc.lower() for w in ['up', 'growth', 'gain', 'rise', 'bull', '호재', '상승', '최고', '매수']):
                    status = "🟢 호재 (긍정)"
                elif any(w in title.lower() or w in desc.lower() for w in ['down', 'fall', 'loss', 'drop', 'bear', '악재', '하락', '우려', '매도']):
                    status = "🔴 악재 (부정)"
                    
                news_list.append({"title": title, "link": link, "desc": desc, "status": status, "summary": summary_text})
    except:
        pass
    return news_list

# --- 🖥️ 홈페이지 레이아웃 시작 ---
st.set_page_config(page_title="나만의 주식 AI 분석기", layout="wide")
st.title("📊🕒 AI 실시간 주가 및 시장 뉴스 대시보드")

if "favorites" not in st.session_state:
    st.session_state.favorites = ["005930.KS", "TSLA", "NVDA"] 
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("⚙️ 분석 설정")

# 즐겨찾기 목록
st.sidebar.subheader("⭐ 즐겨찾는 종목")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        if st.sidebar.button(f"⭐ {fav}", key=f"fav_{fav}", use_container_width=True):
            st.session_state.selected_stock = fav

default_stock = st.session_state.get("selected_stock", "005930.KS")
my_stock = st.sidebar.text_input("1. 종목 코드 입력 (모든 종목 가능)", value=default_stock).upper().strip()

if my_stock in st.session_state.favorites:
    if st.sidebar.button("❌ 즐겨찾기에서 제거", use_container_width=True):
        st.session_state.favorites.remove(my_stock)
        st.rerun()
else:
    if st.sidebar.button("➕ 즐겨찾기에 추가", use_container_width=True):
        st.session_state.favorites.append(my_stock)
        st.rerun()

months_ago = st.sidebar.slider("2. 학습 기간 설정 (개월)", min_value=3, max_value=36, value=14)

st.sidebar.markdown("---")
run_button = st.sidebar.button("종합 시장 분석 시작 🔥", use_container_width=True)

# 최근 검색 기록
st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        if st.sidebar.button(f"🕒 {hist}", key=f"hist_{hist}", use_container_width=True):
            st.session_state.selected_stock = hist
            st.rerun()

# --- 🚀 메인 작동 분석부 ---
if not run_button and "selected_stock" not in st.session_state:
    st.info(f"💡 왼쪽 메뉴에 아무 종목 코드나 넣고 버튼을 누르세요! (국내 주식은 뒤에 .KS / 미국 주식은 AAPL, TSLA 등 티커 입력)")
else:
    if "selected_stock" in st.session_state and not run_button:
        my_stock = st.session_state.selected_stock
        
    if my_stock not in st.session_state.history:
        st.session_state.history.insert(0, my_stock)
        st.session_state.history = st.session_state.history[:5]

    if "selected_stock" in st.session_state:
        del st.session_state.selected_stock

    # 🌟 실시간으로 전 세계 주식 마켓에서 진짜 기업 이름 조회
    with st.spinner("종목 정보를 실시간 조회 중..."):
        stock_display_name = get_exact_stock_name(my_stock)
    
    tab1, tab2 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스 요약"])
    
    with tab1:
        with st.spinner("AI가 데이터를 수집하고 학습하는 중입니다..."):
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
            
            data = yf.download(my_stock, start=start_date, end=end_date)
            
            if len(data) < 30:
                st.error("데이터가 너무 부족합니다. 종목 코드를 다시 확인해 주세요. (예: 삼성전자는 005930.KS)")
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
                    st.caption(f"📅 실시간 분석 기간 : {start_date} ~ {end_date} ({months_ago}개월)")
                    st.metric(label="🎯 업그레이드 AI 정확도", value=f"{accuracy * 100:.2f}%")
                    
                    tomorrow_pred = ai_model.predict(X_today)
                    current_rsi = df['RSI'].iloc[-1]
                    
                    if tomorrow_pred[0] == 1:
                        st.success(f"🔮 AI 최종 판단 : **[ 상승 예상 📈 ]** 내일 {stock_display_name} 주가는 오를 확률이 높습니다.")
                    else:
                        st.error(f"🔮 AI 최종 판단 : **[ 하락 예상 📉 ]** 내일 {stock_display_name} 주가는 떨어질 확률이 높습니다.")
                    
                    st.markdown("##### 💡 기술적 매매 경향 참고")
                    if current_rsi >= 70:
                        st.warning(f"현재 RSI 지표가 **{current_rsi:.1f}**로 과매수 구간입니다. **[단기 매도 포인트]**를 검토할 수 있습니다.")
                    elif current_rsi <= 30:
                        st.info(f"현재 RSI 지표가 **{current_rsi:.1f}**로 과매도 구간입니다. 단기 반등 매수 경향이 강해질 수 있습니다.")
                    else:
                        st.write(f"현재 RSI 지표는 **{current_rsi:.1f}**로 안정적인 흐름입니다.")
                
                with col2:
                    st.subheader(f"📈 {stock_display_name} 주가 및 이동평균선 흐름")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(data['Close'], label='Current Price', color='blue', linewidth=2)
                    ax.plot(data['MA5'], label='5-Day Line', color='green', linestyle=':')
                    ax.plot(data['MA20'], label='20-Day Line', color='orange', linestyle='--')
                    ax.legend(loc='upper left')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    
    with tab2:
        st.subheader(f"📰 {stock_display_name} ({my_stock}) 관련 실시간 글로벌 시장 뉴스 분석")
        
        summarizer = load_summary_model()
        
        with st.spinner("내장 인공지능(AI)이 실시간 뉴스를 읽고 요약하는 중입니다..."):
            news_data = get_stock_news_ai(my_stock)
            
            if not news_data:
                st.warning("현재 해당 종목의 최신 글로벌 뉴스가 존재하지 않거나 가져올 수 없습니다.")
            else:
                for news in news_data:
                    with st.container():
                        st.markdown(f"### [{news['status']}] [{news['title']}]({news['link']})")
                        
                        try:
                            if len(news['summary']) > 30:
                                ai_summary = summarizer(news['summary'], max_length=50, min_length=10, do_sample=False)[0]['summary_text']
                            else:
                                ai_summary = news['summary']
                        except:
                            ai_summary = news['summary']
                            
                        st.success(f"🤖 **내장 AI 기사 실시간 요약:** {ai_summary}")
                        st.caption(f"🔗 *원문 출처 및 내용:* {news['desc']}")
                        st.markdown("---")