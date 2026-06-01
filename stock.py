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

# --- [기존 기능 유지] 기존 함수들은 그대로 유지합니다 ---
def translate_text(text, target_lang="en"):
    if not text: return ""
    try:
        base_url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q="
        url = base_url + urllib.parse.quote(text)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return "".join([sentence[0] for sentence in result[0] if sentence[0]])
    except: pass
    return text

def get_current_usd_krw():
    try:
        usd_krw = yf.Ticker("KRW=X")
        exchange_rate = usd_krw.history(period="1d")['Close'].iloc[-1]
        return float(exchange_rate)
    except: return 1350.0

def get_market_simple_label(ticker_symbol, exchange_name=""):
    ticker_symbol = ticker_symbol.upper()
    exchange_name = exchange_name.upper()
    if ticker_symbol.endswith('.KS'): return "코스피"
    elif ticker_symbol.endswith('.KQ'): return "코스닥"
    elif 'NASDAQ' in exchange_name or 'NMS' in exchange_name or 'NGM' in exchange_name: return "나스닥"
    elif 'NYSE' in exchange_name or 'NYQ' in exchange_name: return "NYSE"
    return "해외증시"

def get_related_stock_list_advanced(search_keyword):
    search_keyword = search_keyword.strip()
    if not search_keyword: return []
    if search_keyword.isdigit() and len(search_keyword) == 6:
        return [{"symbol": search_keyword + ".KS", "display_name": f"국내 종목 코드 입력 [{search_keyword} / 코스피]", "shortname": "국내주식", "country": "한국"}]
    clean_keyword = search_keyword.replace("주식회사", "").replace("(주)", "").replace(" ", "")
    stock_options = []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(clean_keyword)}&quotesCount=15"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            for q in res.json().get('quotes', []):
                symbol = q.get('symbol', '')
                if q.get('quoteType') == 'EQUITY' or ('.KS' in symbol or '.KQ' in symbol or len(symbol) <= 5):
                    raw_name = q.get('shortname') or q.get('longname') or symbol
                    kr_name = translate_text(raw_name, target_lang="ko") if raw_name else raw_name
                    exch = q.get('exchange', '')
                    market_label = get_market_simple_label(symbol, exch)
                    pure_code = symbol.split('.')[0] if '.' in symbol else symbol
                    display_label = f"{kr_name} [{pure_code} / {market_label}]"
                    country_group = "한국" if (market_label in ["코스피", "코스닥"]) else "미국 및 해외"
                    stock_options.append({"symbol": symbol, "display_name": display_label, "shortname": kr_name, "country": country_group})
    except: pass
    return stock_options

def detect_market_info(ticker_symbol, info_data):
    ticker_symbol = ticker_symbol.upper()
    if ticker_symbol.endswith('.KS'): return "대한민국 🇰🇷", "KOSPI (코스피)"
    elif ticker_symbol.endswith('.KQ'): return "대한민국 🇰🇷", "KOSDAQ (코스닥)"
    exchange = info_data.get('exchange', '').upper()
    if 'NASDAQ' in exchange or 'NGM' in exchange or 'NMS' in exchange: return "미국 🇺🇸", "NASDAQ (나스닥)"
    elif 'NYQ' in exchange or 'NYSE' in exchange: return "미국 🇺🇸", "NYSE / S&P 500"
    return "글로벌 마켓 🌐", "해외 주요 증시"

def get_stock_news_safe(ticker_symbol):
    news_list = []
    try:
        yahoo_news = yf.Ticker(ticker_symbol).news
        if yahoo_news:
            for article in yahoo_news[:4]:
                title = article.get('title') or '실시간 속보'
                link = article.get('link') or '#'
                pub_name = article.get('publisher') or '금융 채널'
                summary_text = article.get('summary') or title
                status = "😐 중립"
                if any(w in title.lower() for w in ['up', 'growth', 'gain', 'rise', '상승', '호재']): status = "🟢 호재"
                elif any(w in title.lower() for w in ['down', 'fall', 'loss', 'drop', 'bear', '하락', '악재']): status = "🔴 악재"
                news_list.append({"title": title, "link": link, "publisher": pub_name, "status": status, "summary": summary_text})
    except: pass
    return news_list

def save_prediction(ticker, name, pred_text, current_price):
    today_str = datetime.today().strftime('%Y-%m-%d')
    new_data = pd.DataFrame([{"예측일자": today_str, "종목코드": ticker, "종목명": name, "AI예측": pred_text, "예측시점가격": float(current_price), "실제결과": "대기중", "적중여부": "⏳ 대기"}])
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if not ((df['예측일자'] == today_str) & (df['종목코드'] == ticker)).any():
                pd.concat([df, new_data], ignore_index=True).to_csv(HISTORY_FILE, index=False)
        except: pass
    else: new_data.to_csv(HISTORY_FILE, index=False)

def update_prediction_results():
    if not os.path.exists(HISTORY_FILE): return pd.DataFrame()
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
        if updated: df.to_csv(HISTORY_FILE, index=False)
        return df.sort_index(ascending=False)
    except: return pd.DataFrame()

# --- 🖥️ Streamlit 프레임 구성 ---
st.set_page_config(page_title="나만의 주식 AI 분석기", layout="wide")
st.title("📊🕒 AI 실시간 주가 및 시장 뉴스 대시보드")

if "favorites_dict" not in st.session_state: st.session_state.favorites_dict = {"005930.KS": "삼성전자", "TSLA": "테슬라", "NVDA": "엔비디아"}
if "search_term" not in st.session_state: st.session_state.search_term = "삼성"
if "selected_ticker" not in st.session_state: st.session_state.selected_ticker = "005930.KS"
if "fallback_name" not in st.session_state: st.session_state.fallback_name = "삼성전자"

query_params = st.query_params
if "jump_tk" in query_params:
    st.session_state.selected_ticker = query_params["jump_tk"]
    st.session_state.fallback_name = query_params.get("jump_name", "선택종목")
    st.session_state.search_term = query_params.get("jump_name", "선택종목")
    st.query_params.clear()
    st.rerun()

st.sidebar.header("⚙️ 분석 설정")
search_input = st.sidebar.text_input("🔍 검색할 기업 이름 입력", value=st.session_state.search_term).strip()
if search_input != st.session_state.search_term:
    st.session_state.search_term = search_input
    related = get_related_stock_list_advanced(search_input)
    if related:
        st.session_state.selected_ticker = related[0]['symbol']
        st.session_state.fallback_name = related[0]['shortname']
    st.rerun()

st.sidebar.markdown("---")
chart_period = st.sidebar.radio("📅 차트 보기 설정", ["일봉 (Daily)", "주봉 (Weekly)", "월봉 (Monthly)"])
months_ago = st.sidebar.slider("⏰ AI 학습 기간 설정 (개월)", min_value=3, max_value=36, value=14)

st.sidebar.subheader("⭐ 내 즐겨찾기 목록")
for code, name in st.session_state.favorites_dict.items():
    if st.sidebar.button(f"📌 {name} ({code})", key=f"fav_{code}", use_container_width=True):
        st.session_state.search_term = name
        st.session_state.selected_ticker = code
        st.session_state.fallback_name = name
        st.rerun()

related_stocks = get_related_stock_list_advanced(st.session_state.search_term)
if related_stocks:
    st.markdown("### 🔍 연관 기업 종목 선택 목록")
    korean_group = [s for s in related_stocks if s['country'] == "한국"]
    if korean_group:
        st.markdown("##### 🇰🇷 대한민국 자산")
        cols = st.columns(4)
        for i, stock in enumerate(korean_group[:4]):
            with cols[i]:
                if st.button(f"🏢 {stock['display_name']}", key=f"kr_{stock['symbol']}", use_container_width=True):
                    st.session_state.selected_ticker = stock['symbol']; st.session_state.fallback_name = stock['shortname']; st.rerun()
    global_group = [s for s in related_stocks if s['country'] == "미국 및 해외"]
    if global_group:
        st.markdown("##### 🇺🇸 미국 및 글로벌 자산")
        cols = st.columns(4)
        for i, stock in enumerate(global_group[:4]):
            with cols[i]:
                if st.button(f"🏢 {stock['display_name']}", key=f"gl_{stock['symbol']}", use_container_width=True):
                    st.session_state.selected_ticker = stock['symbol']; st.session_state.fallback_name = stock['shortname']; st.rerun()
    st.markdown("---")

selected_ticker = st.session_state.selected_ticker
if selected_ticker:
    info_data = yf.Ticker(selected_ticker).info
    raw_name = info_data.get('longName') or info_data.get('shortName') or st.session_state.fallback_name
    current_stock_name = translate_text(raw_name, target_lang="ko")
    country, market_name = detect_market_info(selected_ticker, info_data)
    is_ko = selected_ticker.endswith('.KS') or selected_ticker.endswith('.KQ')
    
    tab1, tab2, tab3 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스", "🎯 AI 예측 성적표"])
    with tab1:
        df = yf.download(selected_ticker, period=f"{months_ago}mo", progress=False)
        if len(df) >= 20:
            # [추가] 이평선 및 RSI 계산 복구
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            delta = df['Close'].diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
            df['RSI'] = 100 - (100 / (1 + (up.ewm(13).mean() / down.ewm(13).mean())))
            
            # AI 예측
            X = df[['Close', 'Volume', 'MA5', 'MA20', 'RSI']].dropna()
            y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)[-len(X):]
            model = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X.iloc[:-1], y[:-1])
            pred = model.predict(X.iloc[[-1]])[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 AI 기술적 분석 보고서")
                st.info(f"📊 대상 : **{current_stock_name}**")
                
                # [추가] 강력 매수 신호 알림
                is_golden = df['MA5'].iloc[-1] > df['MA20'].iloc[-1]
                if pred == 1 and is_golden:
                    st.success("🔥 **강력 매수 신호 포착!** (AI 상승예측 + 골든크로스)")
                
                st.metric("AI 예측 정확도", f"{(accuracy_score(y[:-1], model.predict(X.iloc[:-1]))*100):.2f}%")
                st.write(f"🔮 판단 : {'상승 예상 📈' if pred == 1 else '하락 예상 📉'}")
                save_prediction(selected_ticker, current_stock_name, "상승 예상" if pred==1 else "하락 예상", float(df['Close'].iloc[-1]))
            
            with col2:
                # [복구] 5평, 20평 복구된 라인 차트
                st.subheader("📈 주가 & 이동평균선 추이")
                st.line_chart(df[['Close', 'MA5', 'MA20']])
                # [추가] RSI 차트
                st.subheader("📊 RSI (과매수/과매도)")
                st.line_chart(df[['RSI']])

    with tab2:
        for news in get_stock_news_safe(selected_ticker):
            st.markdown(f"### {news['status']} {translate_text(news['title'], 'ko')}")
            st.write(translate_text(news['summary'], 'ko'))
            st.markdown("---")
            
    with tab3:
        history_df = update_prediction_results()
        if not history_df.empty:
            st.markdown("### 🔥 최근 5회 종합 성적")
            history_df['링크주소'] = history_df.apply(lambda r: f"/?jump_tk={r['종목코드']}&jump_name={urllib.parse.quote(r['종목명'])}", axis=1)
            st.dataframe(history_df[['예측일자', '종목코드', '링크주소', 'AI예측', '적중여부']], column_config={"링크주소": st.column_config.LinkColumn("종목명", display_text=r"^/\?jump_tk=.+&jump_name=(.+)$")}, use_container_width=True)
