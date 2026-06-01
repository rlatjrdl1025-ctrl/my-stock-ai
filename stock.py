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

# --- 💾 저장소 고정락 (Session State) ---
if "favorites" not in st.session_state:
    st.session_state.favorites = ["005930.KS", "TSLA", "NVDA"]
if "history" not in st.session_state:
    st.session_state.history = []
if "input_ticker" not in st.session_state:
    st.session_state.input_ticker = "005930.KS"

st.sidebar.header("⚙️ 분석 설정")

# [⭐ 즐겨찾는 종목] 클릭 시 입력창 값 즉시 변경 후 리런
st.sidebar.subheader("⭐ 내 즐겨찾기 목록")
if st.session_state.favorites:
    for fav in st.session_state.favorites:
        if st.sidebar.button(f"📌 {fav}", key=f"fav_{fav}", use_container_width=True):
            st.session_state.input_ticker = fav
            st.rerun()
else:
    st.sidebar.caption("등록된 즐겨찾기가 없습니다.")

st.sidebar.markdown("---")

# 종목 코드 입력창 (세션 동기화)
my_stock = st.sidebar.text_input("1. 종목 코드 입력", value=st.session_state.input_ticker).upper().strip()
st.session_state.input_ticker = my_stock

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

# 🌟 [요구사항 반영] 최근 검색 기록을 클릭 가능한 버튼으로 구현 🌟
st.sidebar.subheader("📜 최근 검색 기록")
if st.session_state.history:
    for hist in st.session_state.history:
        # 최근 기록 버튼을 누르면 입력창 값이 바뀌면서 대시보드가 즉시 해당 종목으로 자동 전환됨
        if st.sidebar.button(f"🕒 {hist}", key=f"hist_{hist}", use_container_width=True):
            st.session_state.input_ticker = hist
            st.rerun()
else:
    st.sidebar.caption("최근 검색 기록이 없습니다.")

# 분석 시작 버튼 누를 시 히스토리 상단 추가
if run_button and my_stock:
    if my_stock in st.session_state.history:
        st.session_state.history.remove(my_stock)
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
            
            # 주가 데이터 다운로드
            df_raw = yf.download(my_stock, start=start_date, end=end_date)
            
            if len(df_raw) < 20:
                st.error("데이터가 부족하거나 종목 코드가 올바르지 않습니다. (국내 주식은 뒤에 .KS를 붙여주세요)")
            else:
                # 🌟 [에러 완천 차단 핵심] 2중 컬럼 구조(MultiIndex)를 단일 컬럼 구조로 강제 강하 및 복사
                df_raw.columns = df_raw.columns.get_level_values(0)
                df_flat = pd.DataFrame(df_raw.values, columns=df_raw.columns, index=df_raw.index)
                df_flat.index = pd.to_datetime(df_flat.index)
                
                # 기술적 보조지표 선행 계산 (1차원 데이터 기반 안전 처리)
                df_flat['MA5'] = df_flat['Close'].rolling(window=5).mean()
                df_flat['MA20'] = df_flat['Close'].rolling(window=20).mean()
                
                delta = df_flat['Close'].diff()
                up, down = delta.clip(lower=0), -delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                df_flat['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
                df_flat = df_flat.dropna()
                
                # 🌟 라디오 버튼 선택에 따른 완벽한 주차별/월별 샘플링 (에러 전면 차단)
                if "주봉" in chart_period:
                    processed_df = df_flat.resample('W').last().dropna()
                elif "월봉" in chart_period:
                    processed_df = df_flat.resample('ME').last().dropna()
                else:
                    processed_df = df_flat.copy()
                
                # 차트용 데이터 바인딩 (문자열 날짜 인덱스로 완벽 고정)
                chart_df = pd.DataFrame({
                    '현재가': processed_df['Close'].values,
                    '5일선(단기)': processed_df['MA5'].values,
                    '20일선(장기)': processed_df['MA20'].values
                }, index=processed_df.index.strftime('%Y-%m-%d'))
                
                # AI 머신러닝 학습 및 가동부
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
                        st.info(f"📊 종목명 : **{stock_display_name} ({my_stock})**")
                        st.caption(f"📅 조회 주기 : {chart_period}")
                        st.metric(label="🎯 AI 예측 정확도", value=f"{accuracy * 100:.2f}%")
                        
                        if tomorrow_pred[0] == 1:
                            st.success("🔮 AI 판단 : **[ 상승 예상 📈 ]** 다음 주기에는 주가가 오를 확률이 높습니다.")
                        else:
                            st.error("🔮 AI 판단 : **[ 하락 예상 📉 ]** 다음 주기에는 주가가 떨어질 확률이 높습니다.")
                        
                        # 기술적 보조지표 리포트 출력
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
