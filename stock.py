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

# --- 🌐 백엔드 실시간 한글 번역 엔진 ---
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

# --- 🔍 [연관 검색] 입력한 단어가 포함된 종목 리스트를 실시간으로 추출 ---
def get_related_stock_list(search_keyword):
    search_keyword = search_keyword.strip()
    if not search_keyword:
        return []
        
    # 숫자 6자리 코드 직접 입력 시 예외 처리
    if search_keyword.isdigit() and len(search_keyword) == 6:
        return [{"symbol": search_keyword + ".KS", "shortname": "국내 주식 코드 입력"}]
        
    clean_keyword = search_keyword.replace("주식회사", "").replace("(주)", "").replace(" ", "")
    
    stock_options = []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(clean_keyword)}&quotesCount=10"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            quotes = data.get('quotes', [])
            for q in quotes:
                symbol = q.get('symbol', '')
                if q.get('quoteType') == 'EQUITY' or ('.KS' in symbol or '.KQ' in symbol or len(symbol) <= 5):
                    name = q.get('shortname') or q.get('longname') or symbol
                    stock_options.append({"symbol": symbol, "shortname": name})
    except:
        pass
    return stock_options

def get_exact_stock_name(ticker_symbol):
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        info = ticker_data.info
        return info.get('longName') or info.get('shortName') or ticker_symbol
    except:
        return ticker_symbol

def get_stock_news_safe(ticker_symbol):
    news_list = []
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        yahoo_news = ticker_data.news
        if yahoo_news:
            for article in yahoo_news[:4]:
                title = article.get('title') or article.get('content', {}).get('title') or '실시간 속보'
                link = article.get('link') or article.get('content', {}).get('clickThroughUrl') or '#'
                pub_name = article.get('publisher') or article.get('content', {}).get('provider', {}).get('displayName') or '금융 채널'
                summary_text = article.get('summary') or article.get('content', {}).get('summary') or title
                status = "😐 중립"
                if any(w in title.lower() for w in ['up', 'growth', 'gain', 'rise', 'bull', '상승', '호재']): status = "🟢 호재"
                elif any(w in title.lower() for w in ['down', 'fall', 'loss', 'drop', 'bear', '하락', '악재']): status = "🔴 악재"
                news_list.append({"title": title, "link": link, "publisher": pub_name, "status": status, "summary": summary_text})
    except: pass
    return news_list

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
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(HISTORY_FILE, index=False)
        except: new_data.to_csv(HISTORY_FILE, index=False)
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
                chk_end = datetime.today() + timedelta(days=1)
                if chk_start <= datetime.today():
                    stock_df = yf.download(ticker, start=chk_start.strftime('%Y-%m-%d'), end=chk_end.strftime('%Y-%m-%d'), progress=False)
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

# --- 🖥️ 메모리 동기화 및 세션 관리 ---
if "favorites_dict" not in st.session_state:
    st.session_state.favorites_dict = {"005930.KS": "삼성전자", "TSLA": "테슬라", "NVDA": "엔비디아"}
if "history" not in st.session_state: st.session_state.history = []
if "input_query" not in st.session_state: st.session_state.input_query = "삼성"
if "final_confirmed_ticker" not in st.session_state: st.session_state.final_confirmed_ticker = "005930.KS"

# --- 사이드바 레이아웃 ---
st.sidebar.header("⚙️ 분석 설정")

# 1. 기업 이름 검색창
search_input = st.sidebar.text_input("1. 검색할 기업 이름 입력", value=st.session_state.input_query).strip()

# 검색어가 달라지면 세션 갱신
if search_input != st.session_state.input_query:
    st.session_state.input_query = search_input
    st.rerun()

# 실시간 연관 검색 목록 추출
related_stocks = get_related_stock_list(search_input)

# 연관 종목 라디오 버튼 생성 (안전장치 적용)
if related_stocks:
    st.sidebar.markdown("🔍 **연관 검색 결과 목록**")
    options_format = [f"{s['shortname']} ({s['symbol']})" for s in related_stocks]
    selected_option = st.sidebar.radio("원하시는 종목을 선택해 주세요:", options_format)
    target_ticker = selected_option.split("(")[-1].replace(")", "").strip()
else:
    st.sidebar.caption("💡 검색어를 입력하시면 연관 종목이 정렬됩니다.")
    target_ticker = "005930.KS"

st.sidebar.markdown("---")
chart_period = st.sidebar.radio("📅 차트 보기 설정", ["일봉 (Daily)", "주봉 (Weekly)", "월봉 (Monthly)"])
months_ago = st.sidebar.slider("2. AI 학습 기간 설정 (개월)", min_value=3, max_value=36, value=14)
st.sidebar.markdown("---")

# 🌟 최종 확인 실행 버튼
run_button = st.sidebar.button("종합 시장 분석 시작 🔥", use_container_width=True)

if run_button:
    st.session_state.final_confirmed_ticker = target_ticker
    if target_ticker not in st.session_state.history:
        st.session_state.history.insert(0, target_ticker)
        st.session_state.history = st.session_state.history[:5]
    st.rerun()

# 즐겨찾기 제어 (최종 확정된 종목 기준)
active_ticker = st.session_state.final_confirmed_ticker
is_fav = active_ticker in st.session_state.favorites_dict
fav_check = st.sidebar.checkbox("⭐ 현재 분석 종목 즐겨찾기 등록", value=is_fav, key=f"chk_{active_ticker}")

if fav_check and not is_fav:
    raw_name = get_exact_stock_name(active_ticker)
    st.session_state.favorites_dict[active_ticker] = translate_text(raw_name, target_lang="ko")
    st.rerun()
elif not fav_check and is_fav:
    del st.session_state.favorites_dict[active_ticker]
    st.rerun()

# 즐겨찾기 및 히스토리 UI 표출
st.sidebar.subheader("⭐ 내 즐겨찾기 목록")
if st.session_state.favorites_dict:
    for code, name in st.session_state.favorites_dict.items():
        if st.sidebar.button(f"📌 {name} ({code})", key=f"fav_{code}", use_container_width=True):
            st.session_state.input_query = name
            st.session_state.final_confirmed_ticker = code
            st.rerun()

st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        if st.sidebar.button(f"🕒 {hist}", key=f"hist_{hist}", use_container_width=True):
            st.session_state.final_confirmed_ticker = hist
            st.rerun()

# --- 🚀 메인 메커니즘 가동부 ---
if active_ticker:
    raw_stock_name = get_exact_stock_name(active_ticker)
    current_stock_name = translate_text(raw_stock_name, target_lang="ko")
    is_korean_stock = active_ticker.endswith('.KS') or active_ticker.endswith('.KQ')
    
    tab1, tab2, tab3 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스", "🎯 AI 예측 성적표"])
    
    with tab1:
        with st.spinner("데이터 수집 및 인공지능 학습 중..."):
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
            df_raw = yf.download(active_ticker, start=start_date, end=end_date)
            
            if len(df_raw) < 20:
                st.error("종목 데이터를 불러오는 중 일시적인 지연이 발생했습니다. '분석 시작' 버튼을 다시 한 번 눌러주세요.")
            else:
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
                        st.info(f"📊 분석 대상 : **{current_stock_name} ({active_ticker})**")
                        st.metric(label="🎯 AI 내부 검증 정확도", value=f"{accuracy * 100:.2f}%")
                        pred_txt = "상승 예상 📈" if tomorrow_pred[0] == 1 else "하락 예상 📉"
                        if tomorrow_pred[0] == 1: st.success(f"🔮 AI 판단 : **[ {pred_txt} ]** 주가가 오를 확률이 높습니다.")
                        else: st.error(f"🔮 AI 판단 : **[ {pred_txt} ]** 주가가 떨어질 확률이 높습니다.")
                        save_prediction(active_ticker, current_stock_name, pred_txt, latest_close)
                        
                        st.markdown("### 💡 보조지표 종합 진단")
                        latest_ma5 = processed_df['MA5'].iloc[-1]
                        latest_ma20 = processed_df['MA20'].iloc[-1]
                        fmt_close = f"₩{latest_close:,.0f}" if is_korean_stock else f"${latest_close:,.2f}"
                        if latest_ma5 > latest_ma20: st.success(f"🟢 **이동평균선:** 현재 골든크로스 / 정배열 상태입니다. (현재가: {fmt_close})")
                        else: st.error(f"🔴 **이동평균선:** 현재 데드크로스 / 역배열 상태입니다. (현재가: {fmt_close})")
                    
                    with col2:
                        if is_korean_stock:
                            st.subheader(f"📈 {current_stock_name} 주가 추이 그래프 (단위: ₩)")
                            chart_df = pd.DataFrame({
                                '현재가(원화)': np.round(processed_df['Close']),
                                '5일선': np.round(processed_df['MA5']),
                                '20일선': np.round(processed_df['MA20'])
                            }, index=processed_df.index.strftime('%Y-%m-%d'))
                            st.line_chart(chart_df)
                            st.markdown(f"""> **💰 국내 자산 정산 안내:** 현재 종가는 **₩{latest_close:,.0f}** 입니다.""")
                        else:
                            ex_rate = get_current_usd_krw()
                            st.subheader(f"📈 {current_stock_name} 주가 추이 그래프 (단위: $)")
                            chart_df_usd = pd.DataFrame({
                                '현재가(달러)': np.round(processed_df['Close'], 2),
                                '5일선': np.round(processed_df['MA5'], 2),
                                '20일선': np.round(processed_df['MA20'], 2)
                            }, index=processed_df.index.strftime('%Y-%m-%d'))
                            st.line_chart(chart_df_usd)
                            
                            st.subheader(f"🔄 {current_stock_name} 실시간 원화 환산 추이 그래프 (단위: ₩)")
                            chart_df_krw = pd.DataFrame({
                                '원화 환산가(원)': np.round(processed_df['Close'] * ex_rate)
                            }, index=processed_df.index.strftime('%Y-%m-%d'))
                            st.line_chart(chart_df_krw)
                            
                            st.markdown(f"""> **💱 실시간 환율 계산기:** 현재 환율 **1$ = {ex_rate:,.2f}원** 적용 | 달러 종가: **${latest_close:,.2f}** ➡️ 원화 환산 종가: **₩{latest_close*ex_rate:,.0f}**""")
                else: st.warning("데이터가 부족합니다.")
                    
    with tab2:
        st.subheader(f"📰 {current_stock_name} 관련 실시간 속보 피드")
        news_data = get_stock_news_safe(active_ticker)
        if news_data:
            for news in news_data:
                with st.container():
                    st.markdown(f"### {news['status']} [{translate_text(news['title'], 'ko')}]({news['link']})")
                    st.success(f"💬 **본문 요약:** {translate_text(news['summary'], 'ko')}")
                    st.caption(f"🔗 *제공처:* {news['publisher']}")
                    st.markdown("---")
                    
    with tab3:
        st.subheader("🎯 나의 AI 등락 예측 일기장")
        history_df = update_prediction_results()
        if not history_df.empty: st.dataframe(history_df, use_container_width=True, hide_index=True)
