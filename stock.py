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

# --- 🏛️ 국가 및 상장 마켓 간결 분류기 ---
def get_market_simple_label(ticker_symbol, exchange_name=""):
    ticker_symbol = ticker_symbol.upper()
    exchange_name = exchange_name.upper()
    
    if ticker_symbol.endswith('.KS'):
        return "코스피"
    elif ticker_symbol.endswith('.KQ'):
        return "코스닥"
    elif 'NASDAQ' in exchange_name or 'NMS' in exchange_name or 'NGM' in exchange_name:
        return "나스닥"
    elif 'NYSE' in exchange_name or 'NYQ' in exchange_name:
        return "NYSE"
    return "해외증시"

# --- 🔍 [연관 검색 엔진] 한국어 전환 + 종목번호 + 증권구분 + 최대 노출 보강 ---
def get_related_stock_list_advanced(search_keyword):
    search_keyword = search_keyword.strip()
    if not search_keyword:
        return []
        
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
                    
                    upper_sym = symbol.upper()
                    if "NVDA" in upper_sym: kr_name = "엔비디아"
                    elif "TSLA" in upper_sym: kr_name = "테슬라"
                    elif "AAPL" in upper_sym: kr_name = "애플"
                    elif "MSFT" in upper_sym: kr_name = "마이크로소프트"
                    elif "AMZN" in upper_sym: kr_name = "아마존"
                    elif "GOOG" in upper_sym or "GOOGL" in upper_sym: kr_name = "구글"
                    elif "META" in upper_sym: kr_name = "메타"
                    else:
                        translated = translate_text(raw_name, target_lang="ko")
                        kr_name = translated if (translated and translated.strip()) else raw_name
                    
                    exch = q.get('exchange', '')
                    market_label = get_market_simple_label(symbol, exch)
                    pure_code = symbol.split('.')[0] if '.' in symbol else symbol
                    
                    display_label = f"{kr_name} [{pure_code} / {market_label}]"
                    country_group = "한국" if (market_label in ["코스피", "코스닥"]) else "미국 및 해외"
                    
                    stock_options.append({
                        "symbol": symbol,
                        "display_name": display_label,
                        "shortname": kr_name,
                        "country": country_group
                    })
    except:
        pass
    return stock_options

# --- 🏛️ 마켓 정밀 분류기 (우측 보고서용) ---
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

def get_stock_news_safe(ticker_symbol):
    news_list = []
    try:
        yahoo_news = yf.Ticker(ticker_symbol).news
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
    except:
        pass
    return news_list

# --- 💾 AI 예측 기록 시스템 ---
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

# --- 🖥️ Streamlit 프레임 구성 ---
st.set_page_config(page_title="나만의 주식 AI 분석기", layout="wide")
st.title("📊🕒 AI 실시간 주가 및 시장 뉴스 대시보드")

if "favorites_dict" not in st.session_state: st.session_state.favorites_dict = {"005930.KS": "삼성전자", "TSLA": "테슬라", "NVDA": "엔비디아"}
if "search_term" not in st.session_state: st.session_state.search_term = "삼성"
if "selected_ticker" not in st.session_state: st.session_state.selected_ticker = "005930.KS"
if "fallback_name" not in st.session_state: st.session_state.fallback_name = "삼성전자"

# Query Parameter를 사용한 테이블 내부 클릭 주소 연동 처리 기능
query_params = st.query_params
if "jump_tk" in query_params:
    st.session_state.selected_ticker = query_params["jump_tk"]
    st.session_state.fallback_name = query_params.get("jump_name", "선택종목")
    st.session_state.search_term = query_params.get("jump_name", "선택종목")
    st.query_params.clear() # 처리 후 쿼리 파라미터 청소
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

# --- 🏢 메인 상단: 국가별 / 증권 지수별 가로 리스트 섹션 ---
related_stocks = get_related_stock_list_advanced(st.session_state.search_term)

if related_stocks:
    st.markdown("### 🔍 연관 기업 종목 선택 목록")
    
    korean_group = [s for s in related_stocks if s['country'] == "한국"]
    if korean_group:
        st.markdown("##### 🇰🇷 대한민국 자산 목록 (KOSPI / KOSDAQ)")
        max_cols = 4
        for chunk_idx in range(0, len(korean_group), max_cols):
            chunk = korean_group[chunk_idx : chunk_idx + max_cols]
            cols = st.columns(max_cols)
            for i, stock in enumerate(chunk):
                with cols[i]:
                    is_current = (stock['symbol'] == st.session_state.selected_ticker)
                    btn_label = f"🟢 {stock['display_name']}" if is_current else f"🏢 {stock['display_name']}"
                    if st.button(btn_label, key=f"main_kr_{stock['symbol']}", use_container_width=True):
                        st.session_state.selected_ticker = stock['symbol']
                        st.session_state.fallback_name = stock['shortname']
                        st.rerun()
                        
    global_group = [s for s in related_stocks if s['country'] == "미국 및 해외"]
    if global_group:
        st.markdown("##### 🇺🇸 미국 및 글로벌 자산 목록 (NASDAQ / NYSE / S&P 500)")
        max_cols = 4
        for chunk_idx in range(0, len(global_group), max_cols):
            chunk = global_group[chunk_idx : chunk_idx + max_cols]
            cols = st.columns(max_cols)
            for i, stock in enumerate(chunk):
                with cols[i]:
                    is_current = (stock['symbol'] == st.session_state.selected_ticker)
                    btn_label = f"🟢 {stock['display_name']}" if is_current else f"🏢 {stock['display_name']}"
                    if st.button(btn_label, key=f"main_gl_{stock['symbol']}", use_container_width=True):
                        st.session_state.selected_ticker = stock['symbol']
                        st.session_state.fallback_name = stock['shortname']
                        st.rerun()
    st.markdown("---")
else:
    st.warning("연관된 종목이 없습니다. 정확한 기업명을 입력해 주세요.")

# --- 🚀 메인 프레임워크 구동부 ---
selected_ticker = st.session_state.selected_ticker
fallback_name = st.session_state.fallback_name

if selected_ticker:
    info_data = {}
    try:
        info_data = yf.Ticker(selected_ticker).info
    except:
        pass

    raw_stock_name = info_data.get('longName') or info_data.get('shortName') or fallback_name
    translated_name = translate_text(raw_stock_name, target_lang="ko")
    current_stock_name = translated_name if (translated_name and translated_name.strip()) else fallback_name
    
    country, market_name = detect_market_info(selected_ticker, info_data)
    is_korean_stock = selected_ticker.endswith('.KS') or selected_ticker.endswith('.KQ')
    
    is_fav = selected_ticker in st.session_state.favorites_dict
    if st.checkbox("⭐ 현재 선택한 종목을 즐겨찾기에 등록", value=is_fav, key=f"chk_{selected_ticker}"):
        if not is_fav:
            st.session_state.favorites_dict[selected_ticker] = current_stock_name
            st.rerun()
    else:
        if is_fav:
            del st.session_state.favorites_dict[selected_ticker]
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["📈 AI 주가 예측 및 차트", "📰 실시간 시장 뉴스", "🎯 AI 예측 성적표"])
    
    with tab1:
        with st.spinner("마켓 데이터를 연동하는 중..."):
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - pd.DateOffset(months=months_ago)).strftime('%Y-%m-%d')
            
            try:
                df_raw = yf.download(selected_ticker, start=start_date, end=end_date, progress=False)
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
                            st.info(f"📊 분석 대상 : **{current_stock_name} ({selected_ticker})**")
                            
                            st.markdown(f"""
                            * **소속 국가 :** {country}
                            * **상장 시장 :** **{market_name}**
                            """)
                            
                            st.metric(label="🎯 AI 예측 정확도", value=f"{accuracy * 100:.2f}%")
                            pred_txt = "상승 예상 📈" if tomorrow_pred[0] == 1 else "하락 예상 📉"
                            if tomorrow_pred[0] == 1: st.success(f"🔮 AI 판단 : **[ {pred_txt} ]** 주가 상승 확률이 높습니다.")
                            else: st.error(f"🔮 AI 판단 : **[ {pred_txt} ]** 주가 하락 확률이 높습니다.")
                            save_prediction(selected_ticker, current_stock_name, pred_txt, latest_close)
                            
                            st.markdown("### 💡 보조지표 종합 진단")
                            latest_ma5 = processed_df['MA5'].iloc[-1]
                            latest_ma20 = processed_df['MA20'].iloc[-1]
                            fmt_close = f"₩{latest_close:,.0f}" if is_korean_stock else f"${latest_close:,.2f}"
                            if latest_ma5 > latest_ma20: st.success(f"🟢 **이동평균선:** 정배열 골든크로스 상태 (현재가: {fmt_close})")
                            else: st.error(f"🔴 **이동평균선:** 역배열 데드크로스 압력 (현재가: {fmt_close})")
                        # AI 분석 이유 생성 함수
def get_ai_reasoning(df, pred):
    ma5 = df['MA5'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    reason = []
    
    # 이평선 및 RSI를 활용한 간단한 분석 로직
    if pred == 1:
        reason.append("단기 이동평균선이 상승 추세를 보이고 있으며,")
        reason.append("RSI 지표상 과매수 구간이 아니여서 추가 상승 여력이 존재합니다." if rsi < 70 else "RSI 지표가 과매수권이나 매수 강세가 유지되고 있습니다.")
    else:
        reason.append("단기 이동평균선이 장기 이동평균선을 하회하고 있으며,")
        reason.append("RSI 지표상 하락 압력이 지속될 가능성이 높습니다." if rsi > 30 else "RSI가 과매도권에 진입하여 기술적 반등 가능성이 있습니다.")
    return " ".join(reason)
                        with col2:
                            if is_korean_stock:
                                st.subheader(f"📈 {current_stock_name} 주가 및 이동평균선 추이 (₩)")
                                chart_df = pd.DataFrame({
                                    '현재가': np.round(processed_df['Close']),
                                    '5일선(단기)': np.round(processed_df['MA5']),
                                    '20일선(장기)': np.round(processed_df['MA20'])
                                }, index=processed_df.index.strftime('%Y-%m-%d'))
                                st.line_chart(chart_df)
                                st.markdown(f"""> **💰 국내 자산 정산 안내:** 현재 종가는 **₩{latest_close:,.0f}** 입니다.""")
                            else:
                                ex_rate = get_current_usd_krw()
                                st.subheader(f"📈 {current_stock_name} 글로벌 통합 주가 추이 ($)")
                                
                                chart_df_usd = pd.DataFrame({
                                    '현재가(달러)': np.round(processed_df['Close'], 2),
                                    '5일선($)': np.round(processed_df['MA5'], 2),
                                    '20일선($)': np.round(processed_df['MA20'], 2),
                                    '원화 환산가(₩ / 1,000)': np.round((processed_df['Close'] * ex_rate) / 1000)
                                }, index=processed_df.index.strftime('%Y-%m-%d'))
                                st.line_chart(chart_df_usd)
                                
                                st.markdown(f"""
                                > **💱 실시간 달러 시세 및 원화 환산 통합 리포트**
                                > * **현재 달러 종가:** **${latest_close:,.2f}**
                                > * **현재 적용 환율:** 1달러($) = **{ex_rate:,.2f}원**
                                > * **🔥 최종 원화 환산 가격:** 약 **₩{latest_close * ex_rate:,.0f}**
                                """)
                    else: st.warning("데이터가 부족합니다.")
                else: st.warning("데이터가 부족합니다.")
            except: st.error("해당 종목의 마켓 데이터를 가져오는 데 실패했습니다.")
                    
    with tab2:
        st.subheader(f"📰 {current_stock_name} 관련 실시간 속보 피드")
        try:
            news_data = get_stock_news_safe(selected_ticker)
            if news_data:
                for news in news_data:
                    with st.container():
                        st.markdown(f"### {news['status']} [{translate_text(news['title'], 'ko')}]({news['link']})")
                        st.success(f"💬 **본문 요약:** {translate_text(news['summary'], 'ko')}")
                        st.caption(f"🔗 *제공처:* {news['publisher']}")
                        st.markdown("---")
            else: st.info("현재 수집된 실시간 뉴스가 없습니다.")
        except: st.error("뉴스를 불러오는 중 오류가 발생했습니다.")
                    
    with tab3:
        st.subheader("🎯 나의 AI 등락 예측 일기장 및 성적표")
        try:
            history_df = update_prediction_results()
            if not history_df.empty:
                
                # [기존 기능 유지] 최근 5일 적중여부 종합 성공률 연산 시스템
                resolved_df = history_df[history_df['적중여부'].isin(["⭕ 적중", "❌ 실패"])]
                if not resolved_df.empty:
                    recent_5 = resolved_df.head(5)
                    correct_5 = len(recent_5[recent_5['적중여부'] == "⭕ 적중"])
                    total_5 = len(recent_5)
                    win_rate_5 = (correct_5 / total_5) * 100 if total_5 > 0 else 0
                    
                    st.markdown(f"""
                    ### 🔥 최근 5회 종합 성적표
                    * **최근 5일 판정 상태:** 총 **{total_5}회** 중 **{correct_5}회 적중**
                    * **최근 5일 단기 승률:** **{win_rate_5:.1f}%**
                    """)
                    st.markdown("---")

                # 🌟 [요구사항 전격 반영] 임시 단추 세트 완전 삭제 및 표 내부 종목명 하이퍼링크 다이렉트 바인딩 🌟
                history_df['소속시장'] = history_df['종목코드'].apply(lambda x: "한국 (KOSPI/KOSDAQ)" if (x.endswith('.KS') or x.endswith('.KQ')) else "미국 및 해외")
                
                # 표 내부의 종목명 텍스트를 클릭하면 메인 세션이 리셋되도록 내부 파라미터 URL 생성 주입
                # (기존 테이블 구조와 완벽히 동일하며, 글자 클릭 시 분석 페이지로 순간 이동합니다.)
                history_df['링크주소'] = history_df.apply(lambda r: f"/?jump_tk={r['종목코드']}&jump_name={urllib.parse.quote(r['종목명'])}", axis=1)

                kr_history = history_df[history_df['소속시장'] == "한국 (KOSPI/KOSDAQ)"].copy()
                us_history = history_df[history_df['소속시장'] == "미국 및 해외"].copy()
                
                if not kr_history.empty:
                    st.markdown("##### 🇰🇷 대한민국 (코스피 / 코스닥) 예측 기록")
                    # '종목명' 컬럼에 하이퍼링크 주소를 바인딩하여 표 형태를 완벽히 유지
                    st.dataframe(
                        kr_history[['예측일자', '종목코드', '링크주소', 'AI예측', '예측시점가격', '실제결과', '적중여부']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "링크주소": st.column_config.LinkColumn("종목명", display_text=r"^/\?jump_tk=.+&jump_name=(.+)$")
                        }
                    )
                    
                if not us_history.empty:
                    st.markdown("##### 🇺🇸 미국 및 글로벌 (NASDAQ / NYSE / S&P 500) 예측 기록")
                    st.dataframe(
                        us_history[['예측일자', '종목코드', '링크주소', 'AI예측', '예측시점가격', '실제결과', '적중여부']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "링크주소": st.column_config.LinkColumn("종목명", display_text=r"^/\?jump_tk=.+&jump_name=(.+)$")
                        }
                    )
                    
            else: st.info("아직 누적된 실전 예측 기록이 없습니다.")
        except: st.error("예측 일기장을 불러오는 중 오류가 발생했습니다.")
