import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
import urllib.parse
import requests

# --- 🔍 [기능 추가] 기업 이름으로 주식 코드(티커)를 실시간 역추적하는 엔진 ---
def search_ticker_by_name(search_keyword):
    search_keyword = search_keyword.strip()
    if not search_keyword:
        return "005930.KS"
        
    # 만약 이미 코드 형태(숫자이거나 영어만 있는 경우)라면 역추적 건너뜀
    if search_keyword.replace('.', '').isalnum() and not any(ord(c) >= 12593 for c in search_keyword):
        return search_keyword
        
    # 한국인들이 자주 쓰는 대표 종목 초고속 매핑 사전
    KOREAN_NAME_MAP = {
        "삼성전자": "005930.KS", "삼성": "005930.KS",
        "카카오": "035720.KS",
        "현대차": "005380.KS", "현대자동차": "005380.KS",
        "SK하이닉스": "000660.KS", "하이닉스": "000660.KS",
        "네이버": "035420.KS", "NAVER": "035420.KS",
        "포스코": "005490.KS", "포스코홀딩스": "005490.KS",
        "테슬라": "TSLA", "엔비디아": "NVDA", "애플": "AAPL", "마이크로소프트": "MSFT", "구글": "GOOGL"
    }
    
    if search_keyword in KOREAN_NAME_MAP:
        return KOREAN_NAME_MAP[search_keyword]
        
    try:
        # 사전 외의 다른 모든 종목은 야후 자동완성 API를 통해 실시간 인터넷 검색 추적
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(search_keyword)}&quotesCount=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get('quotes'):
                return data['quotes'][0]['symbol']
    except:
        pass
    return search_keyword

# --- 🔍 실시간 종목 이름 자동 표시 함수 ---
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

# --- 🌐 무료 한글 번역 엔진 ---
def translate_to_korean(text):
    if not text:
        return ""
    try:
        base_url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ko&dt=t&q="
        url = base_url + urllib.parse.quote(text)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return "".join([sentence[0] for sentence in result[0] if sentence[0]])
    except:
        pass
    return text

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

if "favorites" not in st.session_state:
    st.session_state.favorites = ["005930.KS", "TSLA", "NVDA"]
if "history" not in st.session_state:
    st.session_state.history = []
if "input_query" not in st.session_state:
    st.session_state.input_query = "삼성전자"

st.sidebar.header("⚙️ 분석 설정")

# [⭐ 즐겨찾는 종목]
st.sidebar.subheader("⭐ 내 즐겨찾기 목록")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        if st.sidebar.button(f"📌 {fav}", key=f"fav_{fav}", use_container_width=True):
            st.session_state.input_query = fav
            st.rerun()

st.sidebar.markdown("---")

# 🌟 종목 이름 또는 코드 입력창 (한글 이름 입력 완벽 지원)
search_input = st.sidebar.text_input("1. 종목 이름 또는 코드 입력", value=st.session_state.input_query).strip()
st.session_state.input_query = search_input

# 입력된 텍스트를 실시간 시스템 코드로 변환
with st.spinner("종목 검색 엔진 가동 중..."):
    my_stock = search_ticker_by_name(search_input).upper()

# 즐겨찾기 체크박스 제어 (변환된 시스템 코드를 기준으로 저장)
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
st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        if st.sidebar.button(f"🕒 {hist}", key=f"hist_{hist}", use_container_width=True):
            st.session_state.input_query = hist
            st.rerun()
else:
    st.sidebar.caption("최근 검색 기록이 없습니다.")

if run_button and my_stock:
    if my_stock not in st.session_state.history:
        st.session_state.history.insert(0, my_stock)
        st.session_state.history = st.session_state.history[:5]
    st.rerun()

# --- 🚀 메인 작동부 ---
if my_stock:
    stock_display_name = get_exact_stock_name(my_stock)
    tab1, tab2 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스"])
    
    with tab1:
        with st.spinner("데이터를 안전하게 정제하고 학습하는 중..."):
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
            
            df_raw = yf.download(my_stock, start=start_date, end=end_date)
            
            if len(df_raw) < 20:
                st.error("종목을 찾을 수 없거나 데이터가 부족합니다. 한글 이름을 정확히 입력하거나 미국 주식 티커(예: TSLA)를 입력해 주세요.")
            else:
                df_raw.columns = df_raw.columns.get_level_values(0)
                df_flat = pd.DataFrame(df_raw.values, columns=df_raw.columns, index=df_raw.index)
                df_flat.index = pd.to_datetime(df_flat.index)
                
                df_flat['MA5'] = df_flat['Close'].rolling(window=5).mean()
                df_flat['MA20'] = df_flat['Close'].rolling(window=20).mean()
                
                delta = df_flat['Close'].diff()
                up, down = delta.clip(lower=0), -delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                df_flat['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
                df_flat = df_flat.dropna()
                
                if "주봉" in chart_period:
                    processed_df = df_flat.resample('W').last().dropna()
                elif "월봉" in chart_period:
                    processed_df = df_flat.resample('ME').last().dropna()
                else:
                    processed_df = df_flat.copy()
                
                chart_df = pd.DataFrame({
                    '현재가': processed_df['Close'].values,
                    '5일선(단기)': processed_df['MA5'].values,
                    '20일선(장기)': processed_df['MA20'].values
                }, index=processed_df.index.strftime('%Y-%m-%d'))
                
                X = processed_df[['Close', 'Volume', 'MA5', 'MA20', 'RSI']]
                processed_df['Target'] = np.where(processed_df['Close'].shift(-1) > processed_df['Close'], 1, 0)
                y = processed_df['Target']
                
                if len(X) > 5:
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
                        st.info(f"📊 검색된 종목 : **{stock_display_name} ({my_stock})**")
                        st.caption(f"📅 조회 주기 : {chart_period}")
                        st.metric(label="🎯 AI 예측 정확도", value=f"{accuracy * 100:.2f}%")
                        
                        if tomorrow_pred[0] == 1:
                            st.success("🔮 AI 판단 : **[ 상승 예상 📈 ]** 다음 주기에는 주가가 오를 확률이 높습니다.")
                        else:
                            st.error("🔮 AI 판단 : **[ 하락 예상 📉 ]** 다음 주기에는 주가가 떨어질 확률이 높습니다.")
                        
                        st.markdown("### 💡 보조지표 종합 진단")
                        latest_close = processed_df['Close'].iloc[-1]
                        latest_ma5 = processed_df['MA5'].iloc[-1]
                        latest_ma20 = processed_df['MA20'].iloc[-1]
                        latest_rsi = processed_df['RSI'].iloc[-1]
                        
                        if latest_ma5 > latest_ma20:
                            st.success(f"🟢 **이동평균선:** 현재 단기 이평선이 장기 이평선 위에 있는 **[골든크로스 / 정배열]** 상태입니다. 상승 추세입니다.")
                        else:
                            st.error(f"🔴 **이동평균선:** 현재 단기 이평선이 장기 이평선 아래에 있는 **[데드크로스 / 역배열]** 상태입니다. 하방 압력에 주의하세요.")
                        
                        if latest_rsi >= 70:
                            st.warning(f"⚠️ **RSI 심리도:** 현재 RSI가 **{latest_rsi:.1f}**로 **[과매수 과열 구간]**입니다. 추격 매수는 위험할 수 있습니다.")
                        elif latest_rsi <= 30:
                            st.info(f"🔵 **RSI 심리도:** 현재 RSI가 **{latest_rsi:.1f}**로 **[과매도 공포 구간]**입니다. 단기 반등을 기대해볼 수 있습니다.")
                        else:
                            st.write(f"😐 **RSI 심리도:** 현재 RSI 지표는 **{latest_rsi:.1f}**로 안정적인 박스권 영역입니다.")
                    
                    with col2:
                        st.subheader(f"📈 {stock_display_name} 통합 추이 그래프")
                        st.line_chart(chart_df)
                else:
                    st.warning("데이터가 부족하여 분석을 진행할 수 없습니다. 기간 설정을 조금 더 늘려주세요.")
                    
    with tab2:
        st.subheader(f"📰 {stock_display_name} 관련 실시간 속보 피드 (한글 번역)")
        with st.spinner("뉴스를 실시간으로 한글로 번역하는 중..."):
            news_data = get_stock_news_safe(my_stock)
            
            if not news_data:
                st.warning("현재 수집된 실시간 시장 뉴스가 없습니다.")
            else:
                for news in news_data:
                    with st.container():
                        ko_title = translate_to_korean(news['title'])
                        ko_summary = translate_to_korean(news['summary'])
                        
                        st.markdown(f"### {news['status']} [{ko_title}]({news['link']})")
                        st.success(f"💬 **본문 요약:** {ko_summary}")
                        st.caption(f"🔗 *제공처:* {news['publisher']} (원문 제목: {news['title']})")
                        st.markdown("---")
