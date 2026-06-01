import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime

# --- 🔍 실시간 종목 이름 자동 조회 함수 ---
def get_exact_stock_name(ticker_symbol):
    KOREAN_STOCK_MAP = {
        "005930.KS": "삼성전자", "035720.KS": "카카오", "005380.KS": "현대차",
        "000660.KS": "SK하이닉스", "035420.KS": "NAVER", "005490.KS": "POSCO홀딩스"
    }
    if ticker_symbol in KOREAN_STOCK_MAP:
        return KOREAN_STOCK_MAP[ticker_symbol]
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        info = ticker_data.info
        name = info.get('longName') or info.get('shortName') or ticker_symbol
        return name
    except:
        return ticker_symbol

# --- 📰 실시간 뉴스 수집 엔진 (에러 차단 버전) ---
def get_stock_news_safe(ticker_symbol):
    news_list = []
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        yahoo_news = ticker_data.news
        if yahoo_news:
            for article in yahoo_news[:4]:
                title = article.get('title') or article.get('content', {}).get('title') or '실시간 시장 속보'
                link = article.get('link') or article.get('content', {}).get('clickThroughUrl') or '#'
                pub_name = article.get('publisher') or article.get('content', {}).get('provider', {}).get('displayName') or '금융 리포트'
                summary_text = article.get('summary') or article.get('content', {}).get('summary') or title
                
                status = "😐 중립"
                if any(w in title.lower() for w in ['up', 'growth', 'gain', 'rise', 'bull', '상승', '호재', '최고']):
                    status = "🟢 호재"
                elif any(w in title.lower() for w in ['down', 'fall', 'loss', 'drop', 'bear', '하락', '악재', '우려']):
                    status = "🔴 악재"
                    
                news_list.append({"title": title, "link": link, "publisher": pub_name, "status": status, "summary": summary_text})
    except:
        pass
    return news_list

# --- 🖥️ 대시보드 설정 ---
st.set_page_config(page_title="나만의 주식 AI 분석기", layout="wide")
st.title("📊🕒 AI 실시간 주가 및 시장 뉴스 대시보드")

# --- 💾 저장소 고정락 ---
if "favorites" not in st.session_state:
    st.session_state.favorites = ["005930.KS", "TSLA", "NVDA"]
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("⚙️ 분석 설정")

# [즐겨찾기 목록] 버튼 클릭 시 해당 종목을 입력창 기본값으로 전달
st.sidebar.subheader("⭐ 내 즐겨찾기 목록")
selected_from_fav = None
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        if st.sidebar.button(f"📌 {fav}", key=f"btn_{fav}", use_container_width=True):
            selected_from_fav = fav
else:
    st.sidebar.caption("등록된 즐겨찾기가 없습니다.")

st.sidebar.markdown("---")

# 1. 종목 코드 입력창
if selected_from_fav:
    my_stock = st.sidebar.text_input("1. 종목 코드 입력", value=selected_from_fav).upper().strip()
else:
    my_stock = st.sidebar.text_input("1. 종목 코드 입력", value="005930.KS").upper().strip()

# 🌟 [요구사항 반영] 즐겨찾기 추가/제거 체크박스 제어 시스템 🌟
is_fav = my_stock in st.session_state.favorites
fav_check = st.sidebar.checkbox("⭐ 이 종목 즐겨찾기 등록", value=is_fav, key=f"chk_{my_stock}")

if fav_check and not is_fav:
    st.session_state.favorites.append(my_stock)
    st.rerun()
elif not fav_check and is_fav:
    st.session_state.favorites.remove(my_stock)
    st.rerun()

# 차트 단위 설정
chart_period = st.sidebar.radio("📅 차트 보기 설정", ["일봉 (Daily)", "주봉 (Weekly)", "월봉 (Monthly)"])
months_ago = st.sidebar.slider("2. AI 학습 기간 설정 (개월)", min_value=3, max_value=36, value=14)

st.sidebar.markdown("---")
run_button = st.sidebar.button("종합 시장 분석 시작 🔥", use_container_width=True)

# 최근 검색 기록 관리
if run_button and my_stock and (not st.session_state.history or st.session_state.history[0] != my_stock):
    if my_stock in st.session_state.history:
        st.session_state.history.remove(my_stock)
    st.session_state.history.insert(0, my_stock)
    st.session_state.history = st.session_state.history[:5]

st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        st.sidebar.caption(f"🕒 {hist}")
else:
    st.sidebar.caption("기록 없음")

# --- 🚀 메인 작동부 ---
if my_stock:
    stock_display_name = get_exact_stock_name(my_stock)
    tab1, tab2 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스"])
    
    with tab1:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
        
        raw_data = yf.download(my_stock, start=start_date, end=end_date)
        
        if len(raw_data) < 15:
            st.error("데이터가 부족하거나 종목 코드가 올바르지 않습니다.")
        else:
            # MultiIndex 데이터 정제 및 동기화 안전장치
            if isinstance(raw_data.index, pd.MultiIndex):
                raw_data.index = raw_data.index.get_level_values(0)
            raw_data.index = pd.to_datetime(raw_data.index)
            
            # 🌟 [차트 깨짐 방지] st.line_chart 전용 리샘플링 데이터 구축
            if "주봉" in chart_period:
                chart_data = raw_data['Close'].resample('W').last()
            elif "월봉" in chart_period:
                chart_data = raw_data['Close'].resample('ME').last()
            else:
                chart_data = raw_data['Close'].copy()
                
            chart_df = pd.DataFrame(chart_data)
            chart_df.columns = ['종가 (Close)']
            
            # AI 머신러닝 학습 알고리즘용 데이터 가공
            raw_data['MA5'] = raw_data['Close'].rolling(window=5).mean()
            raw_data['MA20'] = raw_data['Close'].rolling(window=20).mean()
            delta = raw_data['Close'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            raw_data['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
            
            df = raw_data.dropna().copy()
            X = df[['Close', 'Volume', 'MA5', 'MA20', 'RSI']]
            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            y = df['Target']
            
            X_today = X.iloc[[-1]]
            X, y = X.iloc[:-1], y.iloc[:-1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            ai_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            ai_model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, ai_model.predict(X_test))
            tomorrow_pred = ai_model.predict(X_today)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 AI 분석 보고서")
                st.info(f"📊 종목명 : **{stock_display_name} ({my_stock})**")
                st.caption(f"📅 주기 : {chart_period}")
                st.metric(label="🎯 AI 예측 정확도", value=f"{accuracy * 100:.2f}%")
                
                if tomorrow_pred[0] == 1:
                    st.success("🔮 AI 판단 : **[ 상승 예상 📈 ]** 내일은 주가가 오를 확률이 높습니다.")
                else:
                    st.error("🔮 AI 판단 : **[ 하락 예상 📉 ]** 내일은 주가가 떨어질 확률이 높습니다.")
            
            with col2:
                st.subheader(f"📈 {stock_display_name} 주가 추이 그래프")
                # 🌟 에러가 절대 나지 않는 스트리밀릿 표준 라인 차트 구현
                st.line_chart(chart_df)
                
    with tab2:
        st.subheader(f"📰 {stock_display_name} 관련 실시간 속보 피드")
        news_data = get_stock_news_safe(my_stock)
        
        if not news_data:
            st.warning("현재 수집된 실시간 시장 뉴스가 없습니다.")
        else:
            for news in news_data:
                with st.container():
                    st.markdown(f"### {news['status']} [{news['title']}]({news['link']})")
                    st.info(f"💬 **본문 요약:** {news['summary']}")
                    st.caption(f"🔗 *제공처:* {news['publisher']}")
                    st.markdown("---")
