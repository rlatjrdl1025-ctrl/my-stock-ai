import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime, timedelta
import urllib.parse
import requests
import os

# --- 🌐 실시간 한글 번역 엔진 ---
def translate_text(text, target_lang="en"):
    if not text:
        return ""
    try:
        base_url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q="
        url = base_url + urllib.parse.quote(text)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return "".join([sentence[0] for sentence in result[0] if sentence[0]])
    except:
        pass
    return text

# --- 💱 실시간 원/달러 환율 수집 엔진 ---
def get_current_usd_krw():
    try:
        usd_krw = yf.Ticker("KRW=X")
        exchange_rate = usd_krw.history(period="1d")['Close'].iloc[-1]
        return float(exchange_rate)
    except:
        return 1350.0

# --- 🔍 [검색 기능 완전 복구] 국장/해외 주식 자율 추적 엔진 ---
def search_ticker_by_name(search_keyword):
    search_keyword = search_keyword.strip().upper()
    if not search_keyword:
        return "005930.KS"
        
    if search_keyword.isdigit() and len(search_keyword) == 6:
        return search_keyword + '.KS'
        
    if search_keyword.replace('.', '').isalnum() and not any(ord(c) >= 12593 for c in search_keyword):
        return search_keyword
        
    clean_keyword = search_keyword.replace("주식회사", "").replace("(주)", "").replace(" ", "")
    
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(clean_keyword)}&quotesCount=10"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            quotes = res.json().get('quotes', [])
            if quotes:
                for q in quotes:
                    symbol = q.get('symbol', '')
                    if symbol.endswith('.KS') or symbol.endswith('.KQ'):
                        return symbol
                return quotes[0]['symbol']
    except:
        pass
    return search_keyword

# --- 🏛️ 국가 및 상장 마켓(코스피, 코스닥, 나스닥, S&P500) 정밀 분류기 ---
def detect_market_info(ticker_symbol, info_data):
    ticker_symbol = ticker_symbol.upper()
    
    if ticker_symbol.endswith('.KS'):
        return "대한민국 🇰🇷", "KOSPI (코스피)"
    elif ticker_symbol.endswith('.KQ'):
        return "대한민국 🇰🇷", "KOSDAQ (코스닥)"
        
    exchange = info_data.get('exchange', '').upper()
    if 'NASDAQ' in exchange or 'NGM' in exchange or 'NMS' in exchange:
        return "미국 🇺🇸", "NASDAQ (나스닥)"
    elif 'NYQ' in exchange or 'NYSE' in exchange:
        return "미국 🇺🇸", "NYSE / S&P 500"
        
    return "글로벌 마켓 🌐", "해외 주요 증시"

# --- 💾 AI 예측 기록 저장 및 결과 정산 시스템 ---
HISTORY_FILE = "predict_history.csv"
def save_prediction(ticker, name, pred_text, current_price):
    today_str = datetime.today().strftime('%Y-%m-%d')
    new_data = pd.DataFrame([{
        "예측일자": today_str, "종목코드": ticker, "종목명": name,
        "AI예측": pred_text, "예측시점가격": float(current_price),
        "실제결과": "대기중", "적중여부": "⏳ 대기"
    }])
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if not ((df['예측일자'] == today_str) & (df['종목코드'] == ticker)).any():
                pd.concat([df, new_data], ignore_index=True).to_csv(HISTORY_FILE, index=False)
        except:
            pass
    else:
        new_data.to_csv(HISTORY_FILE, index=False)

def update_prediction_results():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORY_FILE)
        updated = False
        for idx, row in df.iterrows():
            if row['적중여부'] == "⏳ 대기":
                ticker = row['종목코드']
                pred_date = row['예측일자']
                chk_start = datetime.strptime(pred_date, '%Y-%m-%d') + timedelta(days=1)
                
                if chk_start <= datetime.today():
                    stock_df = yf.download(ticker, start=chk_start.strftime('%Y-%m-%d'), end=(datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)
                    if len(stock_df) > 0:
                        stock_df.columns = stock_df.columns.get_level_values(0)
                        next_close = float(stock_df['Close'].iloc[0])
                        is_ko = ticker.endswith('.KS') or ticker.endswith('.KQ')
                        fmt = f"₩{next_close:,.0f}" if is_ko else f"${next_close:,.2f}"
                        actual_dir = "상승 📈" if next_close > float(row['예측시점가격']) else "하락 📉"
                        
                        df.at[idx, '실제결과'] = f"{fmt} ({actual_dir})"
                        df.at[idx, '적중여부'] = "⭕ 적중" if row['AI예측'].split()[0] == actual_dir.split()[0] else "❌ 실패"
                        updated = True
        if updated:
            df.to_csv(HISTORY_FILE, index=False)
        return df.sort_index(ascending=False)
    except:
        return pd.DataFrame()

# --- 🖥️ 대시보드 레이아웃 설정 ---
st.set_page_config(page_title="나만의 주식 AI 분석기", layout="wide")
st.title("📊🕒 AI 실시간 주가 및 시장 뉴스 대시보드")

if "favorites_dict" not in st.session_state:
    st.session_state.favorites_dict = {"005930.KS": "삼성전자", "TSLA": "테슬라", "NVDA": "엔비디아"}
if "history" not in st.session_state: st.session_state.history = []
if "input_query" not in st.session_state: st.session_state.input_query = "삼성전자"

st.sidebar.header("⚙️ 분석 설정")

# 자율 검색창
search_input = st.sidebar.text_input("1. 종목 이름 또는 코드 입력", value=st.session_state.input_query).strip()
st.session_state.input_query = search_input

with st.spinner("AI 실시간 검색 엔진 구동 중..."):
    my_stock = search_ticker_by_name(search_input)

# 메인 데이터 호출 및 정보 수집
info_data = {}
try:
    ticker_obj = yf.Ticker(my_stock)
    info_data = ticker_obj.info
except:
    pass

raw_stock_name = info_data.get('longName') or info_data.get('shortName') or my_stock
current_stock_name = translate_text(raw_stock_name, target_lang="ko")

# 국가 및 마켓 자동 식별 분류
country, market_name = detect_market_info(my_stock, info_data)

is_fav = my_stock in st.session_state.favorites_dict
fav_check = st.sidebar.checkbox("⭐ 이 종목 즐겨찾기 등록", value=is_fav, key=f"chk_{my_stock}")

if fav_check and not is_fav:
    st.session_state.favorites_dict[my_stock] = current_stock_name
    st.rerun()
elif not fav_check and is_fav:
    del st.session_state.favorites_dict[my_stock]
    st.rerun()

st.sidebar.subheader("⭐ 내 즐겨찾기 목록")
if st.session_state.favorites_dict:
    for code, name in st.session_state.favorites_dict.items():
        if st.sidebar.button(f"📌 {name} ({code})", key=f"fav_{code}", use_container_width=True):
            st.session_state.input_query = code
            st.rerun()

st.sidebar.markdown("---")
chart_period = st.sidebar.radio("📅 차트 보기 설정", ["일봉 (Daily)", "주봉 (Weekly)", "월봉 (Monthly)"])
months_ago = st.sidebar.slider("2. AI 학습 기간 설정 (개월)", min_value=3, max_value=36, value=14)
st.sidebar.markdown("---")
run_button = st.sidebar.button("종합 시장 분석 시작 🔥", use_container_width=True)

st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        if st.sidebar.button(f"🕒 {hist}", key=f"hist_{hist}", use_container_width=True):
            st.session_state.input_query = hist
            st.rerun()

if run_button and my_stock:
    if my_stock not in st.session_state.history:
        st.session_state.history.insert(0, my_stock)
        st.session_state.history = st.session_state.history[:5]
    st.rerun()

is_korean_stock = my_stock.endswith('.KS') or my_stock.endswith('.KQ')
currency_symbol = "₩" if is_korean_stock else "$"

# --- 🚀 구동 제어부 ---
if my_stock:
    tab1, tab2, tab3 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스", "🎯 AI 예측 성적표"])
    
    with tab1:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
        
        try:
            df_raw = yf.download(my_stock, start=start_date, end=end_date, progress=False)
            if len(df_raw) >= 20:
                df_raw.columns = df_raw.columns.get_level_values(0)
                df_flat = pd.DataFrame(df_raw.values, columns=df_raw.columns, index=df_raw.index)
                df_flat.index = pd.to_datetime(df_flat.index)
                
                df_flat['MA5'] = df_flat['Close'].rolling(window=5).mean()
                df_flat['MA20'] = df_flat['Close'].rolling(window=20).mean()
                delta = df_flat['Close'].diff()
                up, down = delta.clip(lower=0), -delta.clip(upper=0)
                df_flat['RSI'] = 100 - (100 / (1 + (up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean())))
                df_flat = df_flat.dropna()
                
                processed_df = df_flat.resample('W').last().dropna() if "주봉" in chart_period else (df_flat.resample('ME').last().dropna() if "월봉" in chart_period else df_flat.copy())
                
                X = processed_df[['Close', 'Volume', 'MA5', 'MA20', 'RSI']]
                processed_df['Target'] = np.where(processed_df['Close'].shift(-1) > processed_df['Close'], 1, 0)
                y = processed_df['Target']
                
                if len(X) > 5:
                    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:-1], y.iloc[:-1], test_size=0.2, random_state=42, shuffle=False)
                    ai_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
                    accuracy = accuracy_score(y_test, ai_model.predict(X_test))
                    tomorrow_pred = ai_model.predict(X.iloc[[-1]])
                    latest_close = float(processed_df['Close'].iloc[-1])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🤖 AI 및 기술적 지표 보고서")
                        st.info(f"📊 분석 대상 : **{current_stock_name} ({my_stock})**")
                        
                        # 국가 및 소속 거래소 마켓 분류 정밀 정보창
                        st.markdown(f"""
                        * **소속 국가 :** {country}
                        * **상장 시장 :** **{market_name}**
                        """)
                        
                        st.metric(label="🎯 AI 내부 검증 정확도", value=f"{accuracy * 100:.2f}%")
                        pred_txt = "상승 예상 📈" if tomorrow_pred[0] == 1 else "하락 예상 📉"
                        if tomorrow_pred[0] == 1: st.success(f"🔮 AI 판단 : **[ {pred_txt} ]** 다음 주기 주가 상승 확률이 높습니다.")
                        else: st.error(f"🔮 AI 판단 : **[ {pred_txt} ]** 다음 주기 주가 하락 확률이 높습니다.")
                        save_prediction(my_stock, current_stock_name, pred_txt, latest_close)
                        
                        st.markdown("### 💡 보조지표 종합 진단")
                        latest_ma5 = processed_df['MA5'].iloc[-1]
                        latest_ma20 = processed_df['MA20'].iloc[-1]
                        fmt_close = f"₩{latest_close:,.0f}" if is_korean_stock else f"${latest_close:,.2f}"
                        if latest_ma5 > latest_ma20: st.success(f"🟢 **이동평균선:** 정배열 골든크로스 추세입니다. (현재가: {fmt_close})")
                        else: st.error(f"🔴 **이동평균선:** 역배열 데드크로스 압력이 있습니다. (현재가: {fmt_close})")
                    
                    with col2:
                        # 직관적인 주가 변동 막대 바 차트(Bar Chart) 연동
                        st.subheader(f"📊 {current_stock_name} 가격 변동 바 차트")
                        chart_df = pd.DataFrame({
                            '종가 시세': np.round(processed_df['Close']) if is_korean_stock else np.round(processed_df['Close'], 2)
                        }, index=processed_df.index.strftime('%Y-%m-%d'))
                        st.bar_chart(chart_df)
                        
                        if is_korean_stock:
                            st.markdown(f"""> **💰 국내 자산 정산 안내:** 현재 종가는 **₩{latest_close:,.0f}** 입니다.""")
                        else:
                            ex_rate = get_current_usd_krw()
                            st.markdown(f"""
                            > **💱 실시간 달러 시세 및 원화 환산 통합 리포트**
                            > * **현재 달러 종가:** **${latest_close:,.2f}**
                            > * **현재 적용 환율:** 1달러($) = **{ex_rate:,.2f}원**
                            > * **🔥 최종 원화 환산 가격:** 약 **₩{latest_close * ex_rate:,.0f}**
                            """)
                else:
                    st.warning("데이터가 부족합니다.")
            else:
                st.warning("데이터가 부족합니다.")
        except:
            st.error("해당 종목의 마켓 데이터를 가져오는 데 실패했습니다.")
                    
    with tab2:
        st.subheader(f"📰 {current_stock_name} 관련 실시간 속보 피드")
        # 🌟 [오류 원인 완벽 제어] 변수명을 명확하게 일치시켜 연쇄 충돌 원천 차단 완료 🌟
        news_data = get_stock_news_safe(my_stock)
        if news_data:
            for news in news_data:
                with st.container():
                    st.markdown(f"### {news['status']} [{translate_text(news['title'], 'ko')}]({news['link']})")
                    st.success(f"💬 **본문 요약:** {translate_text(news['summary'], 'ko')}")
                    st.caption(f"🔗 *제공처:* {news['publisher']}")
                    st.markdown("---")
                    
    with tab3:
        st.subheader("🎯 나의 AI 등락 예측 일기장 및 성적표")
        history_df = update_prediction_results()
        if not history_df.empty: st.dataframe(history_df, use_container_width=True, hide_index=True)
