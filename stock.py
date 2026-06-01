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

# --- 📰 실시간 뉴스 수집 엔진 ---
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

# [즐겨찾기 목록] 
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

# 즐겨찾기 체크박스 제어
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

# --- 🚀 메인 작동부 ---
if my_stock:
    stock_display_name = get_exact_stock_name(my_stock)
    tab1, tab2 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스"])
    
    with tab1:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
        
        raw_data = yf.download(my_stock, start=start_date, end=end_date)
        
        if len(raw_data) < 30:
            st.error("데이터가 부족하거나 종목 코드가 올바르지 않습니다.")
        else:
            if isinstance(raw_data.index, pd.MultiIndex):
                raw_data.index = raw_data.index.get_level_values(0)
            raw_data.index = pd.to_datetime(raw_data.index)
            
            # 주기에 따른 데이터 리샘플링 가공
            if "주봉" in chart_period:
                processed_data = raw_data.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
            elif "월봉" in chart_period:
                processed_data = raw_data.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
            else:
                processed_data = raw_data.copy()
                
            processed_data = processed_data.dropna()
            
            # 🌟 [블로그 용어 적용] 기술적 보조지표 계산 연동
            processed_data['5일선(MA5)'] = processed_data['Close'].rolling(window=5).mean()
            processed_data['20일선(MA20)'] = processed_data['Close'].rolling(window=20).mean()
            
            delta = processed_data['Close'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            processed_data['RSI 지표'] = 100 - (100 / (1 + (ema_up / ema_down)))
            
            # 차트용 데이터프레임 빌드 (절대 깨지지 않는 멀티 라인 구조)
            chart_df = pd.DataFrame({
                '현재가 (Close)': processed_data['Close'],
                '5일 이동평균선': processed_data['5일선(MA5)'],
                '20일 이동평균선': processed_data['20일선(MA20)']
            })
            
            # AI 예측 모델 빌드부
            df_ml = processed_data.dropna().copy()
            X = df_ml[['Close', 'Volume', '5일선(MA5)', '20일선(MA20)', 'RSI 지표']]
            df_ml['Target'] = np.where(df_ml['Close'].shift(-1) > df_ml['Close'], 1, 0)
            y = df_ml['Target']
            
            X_today = X.iloc[[-1]]
            X, y = X.iloc[:-1], y.iloc[:-1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            ai_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            ai_model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, ai_model.predict(X_test))
            tomorrow_pred = ai_model.predict(X_today)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 AI 및 기술적 지표 보고서")
                st.info(f"📊 종목명 : **{stock_display_name} ({my_stock})**")
                st.caption(f"📅 조회 주기 : {chart_period}")
                st.metric(label="🎯 AI 예측 정확도", value=f"{accuracy * 100:.2f}%")
                
                if tomorrow_pred[0] == 1:
                    st.success("🔮 AI 판단 : **[ 상승 예상 📈 ]** 다음 주기에는 주가가 오를 확률이 높습니다.")
                else:
                    st.error("🔮 AI 판단 : **[ 하락 예상 📉 ]** 다음 주기에는 주가가 떨어질 확률이 높습니다.")
                
                # 🌟 [용어 해석 보고서 연동]
                st.markdown("### 💡 보조지표 종합 진단")
                latest_close = df_ml['Close'].iloc[-1]
                latest_ma5 = df_ml['5일선(MA5)'].iloc[-1]
                latest_ma20 = df_ml['20일선(MA20)'].iloc[-1]
                latest_rsi = df_ml['RSI 지표'].iloc[-1]
                
                # 1. 이동평균선 정배열 / 역배열 확인
                if latest_ma5 > latest_ma20:
                    st.success(f"🟢 **이동평균선 진단:** 현재 5일선({latest_ma5:,.0f})이 20일선({latest_ma20:,.0f}) 위에 있는 **[골든크로스 / 정배열]** 상태입니다. 단기 상승 추세가 강합니다.")
                else:
                    st.error(f"🔴 **이동평균선 진단:** 현재 5일선({latest_ma5:,.0f})이 20일선({latest_ma20:,.0f}) 아래에 있는 **[데드크로스 / 역배열]** 상태입니다. 신중한 접근이 필요합니다.")
                
                # 2. RSI 심리 지표 분석
                if latest_rsi >= 70:
                    st.warning(f"⚠️ **RSI 과열도 진단:** 현재 RSI가 **{latest_rsi:.1f}**로 **[과매수 구간(70 이상)]**에 진입했습니다. 시장 심리가 지나치게 과열되어 있으니 과도한 추격 매수는 자제하고 단기 매도를 검토할 타이밍입니다.")
                elif latest_rsi <= 30:
                    st.info(f"🔵 **RSI 과열도 진단:** 현재 RSI가 **{latest_rsi:.1f}**로 **[과매도 구간(30 이하)]**에 진입했습니다. 매도세가 과도하여 바닥권일 확률이 높으며, 단기 기술적 반등 및 분할 매수 진입을 고려해볼 수 있습니다.")
                else:
                    st.write(f"😐 **RSI 과열도 진단:** 현재 RSI 지표는 **{latest_rsi:.1f}**로 심리적 과열이나 공포 없이 안정적인 박스권 흐름을 유지하고 있습니다.")
            
            with col2:
                st.subheader(f"📈 {stock_display_name} 통합 추이 그래프")
                # 🌟 일/주/월봉을 눌러도 보조지표선이 절대 깨지지 않고 겹쳐 나오는 멀티 차트 가동
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
