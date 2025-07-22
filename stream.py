import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import numpy as np
import json
import plotly.express as px

st.set_page_config(page_title="통합 데이터 분석 대시보드", layout="wide")  # 반드시 첫 Streamlit 명령

# 컬럼 분할 먼저!
col_left, col_right = st.columns([8, 3])

with col_left:
    st.title("통합 데이터 분석 대시보드")
    # 데이터 로드 (가장 먼저!)
    users = pd.read_csv('https://www.dropbox.com/scl/fi/egugh9elauchrffiwyn1h/users_preprocessed.csv?rlkey=7knpt37y7z0f5qbm9rm416n5a&st=wfak300j&dl=1', parse_dates=['created_at'])

events = pd.read_csv('https://www.dropbox.com/scl/fi/l9bi2dfvlug1j8fy6biuo/events_preprocessed.csv?rlkey=0gxb54oymo1r7gme298jy98wv&st=gq7geawk&dl=1', parse_dates=['created_at'])

# 전체 데이터 사용
filtered_users = users.copy()
filtered_events = events.copy()

# 신규 유저수 (월평균)
if filtered_users.empty:
    avg_new_users = 0
else:
    filtered_users['year_month'] = filtered_users['created_at'].dt.to_period('M')
    monthly_new_users = filtered_users.groupby('year_month')['id'].nunique()
    avg_new_users = int(monthly_new_users.mean()) if len(monthly_new_users) > 0 else 0

# MAU (월평균)
if filtered_events.empty:
    avg_mau = 0
else:
    filtered_events['year_month'] = filtered_events['created_at'].dt.to_period('M')
    monthly_mau = filtered_events.groupby('year_month')['user_id'].nunique()
    avg_mau = int(monthly_mau.mean()) if len(monthly_mau) > 0 else 0

# 이탈률 (월평균)
if filtered_events.empty:
    avg_churn = 0
    churn_rates = []
    months = []
else:
    # 월별 유저 집합 구하기
    filtered_events['year_month'] = filtered_events['created_at'].dt.to_period('M')
    months = sorted(filtered_events['year_month'].unique())
    churn_rates = []
    for i, month in enumerate(months[:-1]):
        this_month_users = set(filtered_events[filtered_events['year_month'] == month]['user_id'])
        next_month = months[i+1]
        next_month_users = set(filtered_events[filtered_events['year_month'] == next_month]['user_id'])
        churned = this_month_users - next_month_users
        churn_rate = len(churned) / len(this_month_users) * 100 if len(this_month_users) > 0 else 0
        churn_rates.append(churn_rate)
    if churn_rates:
        avg_churn = sum(churn_rates) / len(churn_rates)
    else:
        avg_churn = 0

# Carrying Capacity (월별 신규 유저수 / 월별 이탈률)
if filtered_users.empty or filtered_events.empty:
    avg_cc = 0
    cc_list = []
else:
    # 월별 신규 유저수
    monthly_new_users = filtered_users.groupby('year_month')['id'].nunique()
    # 월별 이탈률 (churn_rates, months는 이미 위에서 계산)
    cc_list = []
    for i, month in enumerate(months[:-1]):
        new_users = monthly_new_users.get(month, 0)
        churn = churn_rates[i]
        if churn > 0:
            cc = new_users / churn
            cc_list.append(cc)
    avg_cc = int(sum(cc_list) / len(cc_list)) if cc_list else 0

# 스코어카드에 표시
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("월평균 신규 유저수", f"{avg_new_users:,}", help=f"전체 데이터 기준 월별 평균 신규 가입자 수")
with col2:
    st.metric("월평균 MAU", f"{avg_mau:,}", help=f"전체 데이터 기준 월별 평균 MAU")
with col3:
    st.metric("월평균 이탈률 (%)", f"{avg_churn:.2f}", help=f"전체 데이터 기준 월별 평균 이탈률")
with col4:
    st.metric("월평균 Carrying Capacity", f"{avg_cc:,}", help=f"전체 데이터 기준 월별 신규 유저수 ÷ 월별 이탈률의 평균")

# 탭 구성
user_tab, purchase_tab, product_tab, region_tab = st.tabs([
    "유저 분석", "구매 분석", "제품 분석", "지역/유입 분석"
])

# 1. 월별 MAU 계산
if filtered_events.empty:
    monthly_mau = pd.Series(dtype=int)
else:
    filtered_events['year_month'] = filtered_events['created_at'].dt.to_period('M')
    monthly_mau = filtered_events.groupby('year_month')['user_id'].nunique()

# 2. 월별 신규 유저수
if filtered_users.empty:
    monthly_new_users = pd.Series(dtype=int)
else:
    filtered_users['year_month'] = filtered_users['created_at'].dt.to_period('M')
    monthly_new_users = filtered_users.groupby('year_month')['id'].nunique()

# 붉은색 계열 명도 변수 정의
main_red = 'rgb(220,53,69)'         # 탭 선택 컬러와 동일
mid_red = 'rgb(255,99,132)'
light_red = 'rgb(255,179,186)'
highlight = 'rgb(0,123,255)'  # 파란색(bootstrap blue)
highlight_yellow = 'rgb(255,193,7)'  # 노란색(bootstrap yellow)

# 2. Bar 그래프 생성 (붉은색 계열, 명도 차이)
fig_new_users = go.Figure()
fig_new_users.add_bar(
    x=[str(m) for m in monthly_new_users.index],
    y=monthly_new_users.values,
    name="신규 유저수",
    marker_color=main_red
)
fig_new_users.update_layout(
    title="월별 신규 유저 수",
    xaxis_title="월",
    yaxis_title="신규 유저수",
    height=300
)

# 1. 월별 Carrying Capacity & MAU 복합 그래프 생성 (붉은색 계열, 명도 차이)
highlight = 'rgb(0,123,255)'  # 파란색(bootstrap blue)

highlight_n = 3
churn_arr = np.array(churn_rates)
high_churn_idx = churn_arr.argsort()[-highlight_n:][::-1] if len(churn_arr) >= highlight_n else []
low_churn_idx = churn_arr.argsort()[:highlight_n] if len(churn_arr) >= highlight_n else []
cc_arr = np.array(cc_list)
high_cc_idx = cc_arr.argsort()[-highlight_n:][::-1] if len(cc_arr) >= highlight_n else []
low_cc_idx = cc_arr.argsort()[:highlight_n] if len(cc_arr) >= highlight_n else []
highlight_months = [months[i] for i in np.concatenate([high_churn_idx, low_churn_idx])] if months else []

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_bar(
    x=[str(m) for m in months],
    y=cc_list,
    name="Carrying Capacity",
    marker_color=main_red
)
fig.add_scatter(
    x=[str(m) for m in months],
    y=monthly_mau.values if not isinstance(monthly_mau, pd.Series) or not monthly_mau.empty else [],
    name="MAU",
    mode="lines+markers",
    line=dict(color=light_red, width=3),
    yaxis="y2"
)
# CC 하이라이트 마커(파란색/노란색, 좌측 y축)
fig.add_scatter(
    x=[str(months[i]) for i in high_cc_idx],
    y=[cc_list[i] for i in high_cc_idx],
    mode='markers',
    marker=dict(color=highlight, size=8),
    name='CC High',
    yaxis="y1"
)
fig.add_scatter(
    x=[str(months[i]) for i in low_cc_idx],
    y=[cc_list[i] for i in low_cc_idx],
    mode='markers',
    marker=dict(color=highlight_yellow, size=8),
    name='CC Low',
    yaxis="y1"
)
fig.update_layout(
    title="월별 Carrying Capacity & MAU (하이라이트)",
    xaxis_title="월",
    height=400,
    hovermode='x unified'
)
fig.update_yaxes(title_text="Carrying Capacity", secondary_y=False)
fig.update_yaxes(title_text="MAU", secondary_y=True)

# 월별 이탈률 그래프에 하이라이트 추가
fig_churn = go.Figure()
fig_churn.add_trace(go.Scatter(
    x=[str(m) for m in months],
    y=churn_rates,
    name="이탈률(%)",
    line=dict(color=mid_red, width=3),
    mode='lines+markers'
))
fig_churn.add_scatter(
    x=[str(months[i]) for i in high_churn_idx],
    y=[churn_rates[i] for i in high_churn_idx],
    mode='markers',
    marker=dict(color=highlight, size=8),
    name='Churn High'
)
fig_churn.add_scatter(
    x=[str(months[i]) for i in low_churn_idx],
    y=[churn_rates[i] for i in low_churn_idx],
    mode='markers',
    marker=dict(color=highlight_yellow, size=8),
    name='Churn Low'
)
fig_churn.update_layout(
    title="월별 이탈률 (하이라이트)",
    xaxis_title="월",
    yaxis_title="이탈률(%)",
    height=300
)

# Funnel 단계별 유저 수 집계 예시 (붉은색 계열, 명도 차이)
signup_users = set(users['id'])
first_purchase_users = set(events[events['event_type'] == 'purchase']['user_id'])
repeat_purchase_users = set(
    events[events['event_type'] == 'purchase'].groupby('user_id').filter(lambda x: len(x) > 1)['user_id'].unique()
)

funnel_labels = ['회원가입', '첫구매', '반복구매']
funnel_values = [len(signup_users), len(first_purchase_users), len(repeat_purchase_users)]

fig_funnel = go.Figure(go.Funnel(
    y=funnel_labels,
    x=funnel_values,
    textinfo="value+percent initial",
    marker={"color": [main_red, mid_red, light_red]}
))
fig_funnel.update_layout(
    title="Funnel (회원가입 → 첫구매 → 반복구매)",
    height=400
)

# 2. Streamlit에 표시
with user_tab:
    st.header("유저 분석")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    # with col1:
    #     st.plotly_chart(fig_new_users, use_container_width=True)
    # with col2:
    #     st.plotly_chart(fig_churn, use_container_width=True)

    # 월별 이탈률과 Funnel을 한 줄, 양쪽 좌우에 배치
    col3, col4 = st.columns([3, 2])
    with col3:
        st.plotly_chart(fig_churn, use_container_width=True)
    with col4:
        st.plotly_chart(fig_funnel, use_container_width=True)

    # 이탈 유저 프로파일링 (선택된 하이라이트 구간 기준)
    selected_month = st.selectbox(
        "상세 분석할 하이라이트 구간(월)을 선택하세요",
        options=[str(m) for m in highlight_months],
        index=0
    )
    selected_month = pd.Period(selected_month)
    idx = months.index(selected_month) if selected_month in months else -1
    if idx != -1 and idx+1 < len(months):
        next_month = months[idx+1]
        churn_rate_value = churn_rates[idx]
        st.subheader(f"이탈률 상세 분석: {str(selected_month)} → {str(next_month)}")
        st.markdown(f"**이탈률:** {churn_rate_value:.2f}% (기준: {str(selected_month)} → {str(next_month)})")
        this_month_users = set(filtered_events[filtered_events['year_month'] == selected_month]['user_id'])
        next_month_users = set(filtered_events[filtered_events['year_month'] == next_month]['user_id'])
        churned_users = this_month_users - next_month_users
        churned_profiles = filtered_users[filtered_users['id'].isin(churned_users)]
        profile_colors = [main_red, mid_red, light_red, highlight]
        for i, col in enumerate(['age', 'gender', 'state']):
            if col in churned_profiles.columns:
                counts = churned_profiles[col].value_counts()
                fig_profile = go.Figure()
                fig_profile.add_bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    marker_color=profile_colors[i % len(profile_colors)]
                )
                fig_profile.update_layout(
                    title=f"이탈 유저 {col} 분포",
                    xaxis_title=col,
                    yaxis_title="유저 수",
                    height=300
                )
                st.plotly_chart(fig_profile, use_container_width=True)
                if not counts.empty:
                    top = counts.idxmax()
                    pct = counts.max() / counts.sum() * 100
                    st.info(f"이탈 유저 중 가장 많은 {col}: {top} ({pct:.1f}%)")
        if churned_profiles.empty:
            st.warning("이탈 유저가 없습니다.")

with purchase_tab:
    st.header("구매 분석")

    # 1행: 반복구매율 (월별)
    if filtered_events.empty:
        months_order = []
        repeat_rates = []
    else:
        # 1. 구매 이벤트만 필터링
        purchase_events = filtered_events[
            (filtered_events['event_type'] == 'purchase')
        ].copy()
        purchase_events['year_month'] = purchase_events['created_at'].dt.to_period('M')
        months_order = sorted(purchase_events['year_month'].unique())
        repeat_rates = []
        for month in months_order:
            month_data = purchase_events[purchase_events['year_month'] == month]
            # 'id'가 한 번의 구매를 의미한다고 가정
            user_order_counts = month_data.groupby('user_id')['id'].nunique()
            repeat_count = (user_order_counts >= 2).sum()
            total_count = user_order_counts.shape[0]
            repeat_rate = (repeat_count / total_count * 100) if total_count > 0 else 0
            repeat_rates.append(repeat_rate)

    fig_repeat = go.Figure()
    fig_repeat.add_trace(go.Scatter(
        x=[str(m) for m in months_order],
        y=repeat_rates,
        name="반복구매율(%)",
        line=dict(color=main_red, width=3),
        mode='lines+markers'
    ))
    fig_repeat.update_layout(
        title="월별 반복구매율",
        xaxis_title="월",
        yaxis_title="반복구매율(%)",
        height=350
    )
    st.plotly_chart(fig_repeat, use_container_width=True)

    # 2행: 평균 구매주기 히스토그램, boxplot (가로 배열)
    # 1. 구매 이벤트만 필터링
    if filtered_events.empty:
        all_gaps = []
    else:
        purchase_events = filtered_events[
            (filtered_events['event_type'] == 'purchase')
        ].copy()
        # 2. user_id별 구매 간격(일) 계산
        purchase_events = purchase_events.sort_values(['user_id', 'created_at'])
        purchase_events['prev_date'] = purchase_events.groupby('user_id')['created_at'].shift(1)
        purchase_events['gap_days'] = (purchase_events['created_at'] - purchase_events['prev_date']).dt.days
        all_gaps = purchase_events['gap_days'].dropna().values
    # 히스토그램
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=all_gaps,
        nbinsx=20,
        marker_color=main_red,
        name="구매주기(일)"
    ))
    fig_hist.update_layout(
        title="구매주기 분포 (히스토그램)",
        xaxis_title="구매 간격(일)",
        yaxis_title="빈도",
        height=300
    )
    # 박스플롯
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=all_gaps,
        marker_color=mid_red,
        name="구매주기(일)"
    ))
    fig_box.update_layout(
        title="구매주기 분포 (박스플롯)",
        yaxis_title="구매 간격(일)",
        height=300
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        st.plotly_chart(fig_box, use_container_width=True)

    # 3행: 1회/2회/3회 이상 구매자 비율 (아래 줄)
    # 실제 데이터로 집계
    if 'order_items' not in st.session_state:
        order_items = pd.read_csv('https://www.dropbox.com/scl/fi/1wus8qtn8ig32ee27nupq/order_items_preprocessed.csv?rlkey=l4a5ilw64qcpx8ukhcu79aeex&st=lux24bfy&dl=1')
        st.session_state['order_items'] = order_items
    else:
        order_items = st.session_state['order_items']

    # 전체 기간 기준 user_id별 구매 횟수 집계
    user_purchase_counts = order_items.groupby('user_id').size()
    one_time = (user_purchase_counts == 1).sum()
    two_times = (user_purchase_counts == 2).sum()
    three_plus = (user_purchase_counts >= 3).sum()
    labels = ['1회', '2회', '3회 이상']
    values = [one_time, two_times, three_plus]
    fig_buyer_ratio = go.Figure()
    fig_buyer_ratio.add_bar(
        x=labels,
        y=values,
        marker_color=[main_red, mid_red, light_red]
    )
    fig_buyer_ratio.update_layout(
        title="1회/2회/3회 이상 구매자 비율",
        xaxis_title="구매 횟수",
        yaxis_title="유저 수",
        height=300
    )
    st.plotly_chart(fig_buyer_ratio, use_container_width=True)

    # 반복구매자 프로파일링: 구매주기 짧은 유저 vs 긴 유저 비교
    st.subheader("반복구매자 프로파일링: 구매주기 짧은 유저 vs 긴 유저")
    if 'order_items' not in st.session_state:
        order_items = pd.read_csv('https://www.dropbox.com/scl/fi/1wus8qtn8ig32ee27nupq/order_items_preprocessed.csv?rlkey=l4a5ilw64qcpx8ukhcu79aeex&st=lux24bfy&dl=1')
        st.session_state['order_items'] = order_items
    else:
        order_items = st.session_state['order_items']

    if 'products' not in st.session_state:
        products = pd.read_csv('https://www.dropbox.com/scl/fi/nkg8tz7dn9hsmwav2vi1t/products_preprocessed.csv?rlkey=ajz1rclosfymrfw8ghmllo8vm&st=t4ccuq7k&dl=1')
        st.session_state['products'] = products
    else:
        products = st.session_state['products']
    products['id'] = products['id'].astype(str)

    # 1. user_id별 구매일시 추출
    user_orders = order_items.groupby('user_id')['created_at'].apply(list)
    # 2. user_id별 평균 구매주기 계산
    user_gap = {}
    for uid, dates in user_orders.items():
        if len(dates) < 2:
            continue
        dates_sorted = pd.to_datetime(dates)
        gaps = dates_sorted.sort_values().diff().dropna().days.values
        if len(gaps) > 0:
            user_gap[uid] = np.mean(gaps)
    if len(user_gap) < 10:
        st.info("구매주기 분석을 위한 충분한 반복구매 유저가 없습니다.")
    else:
        gap_series = pd.Series(user_gap)
        short_cut = gap_series.quantile(0.25)
        long_cut = gap_series.quantile(0.75)
        short_users = gap_series[gap_series <= short_cut].index.astype(str)
        long_users = gap_series[gap_series >= long_cut].index.astype(str)
        # 프로필 데이터 로드
        short_profiles = users[users['id'].astype(str).isin(short_users)]
        long_profiles = users[users['id'].astype(str).isin(long_users)]
        # 구매 패턴 데이터
        short_orders = order_items[order_items['user_id'].astype(str).isin(short_users)]
        long_orders = order_items[order_items['user_id'].astype(str).isin(long_users)]
        # 1. 프로필(나이, 성별, 지역, 유입경로)
        profile_cols = ['age', 'state']
        col_short, col_long = st.columns(2)
        with col_short:
            st.markdown(f"**구매주기 짧은 유저 (상위 25%)**: {len(short_users)}명")
            for i, col in enumerate(profile_cols):
                if col == 'age':
                    ages = pd.to_numeric(short_profiles['age'], errors='coerce').dropna().astype(int)
                    bins = [10, 20, 30, 40, 50, 60, 70, 100]
                    labels = ["10대", "20대", "30대", "40대", "50대", "60대", "70대+"]
                    age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
                    age_counts = age_groups.value_counts().sort_index()
                    # 하이라이트 색상 적용
                    bar_colors = [main_red] * len(age_counts)
                    if len(age_counts) > 0:
                        max_idx = age_counts.values.argmax()
                        min_idx = age_counts.values.argmin()
                        bar_colors[max_idx] = highlight  # 파랑
                        bar_colors[min_idx] = highlight_yellow  # 노랑
                    fig = go.Figure()
                    fig.add_bar(x=age_counts.index.astype(str), y=age_counts.values, marker_color=bar_colors)
                    fig.update_layout(title="age 분포 (연령대)", xaxis_title="연령대", yaxis_title="유저 수", height=200)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    counts = short_profiles[col].value_counts().head(10)
                    bar_colors = [main_red] * len(counts)
                    if len(counts) > 0:
                        max_idx = counts.values.argmax()
                        min_idx = counts.values.argmin()
                        bar_colors[max_idx] = highlight
                        bar_colors[min_idx] = highlight_yellow
                    fig = go.Figure()
                    fig.add_bar(x=counts.index.astype(str), y=counts.values, marker_color=bar_colors)
                    fig.update_layout(title=f"{col} 분포", height=200)
                    st.plotly_chart(fig, use_container_width=True)
        with col_long:
            st.markdown(f"**구매주기 긴 유저 (하위 25%)**: {len(long_users)}명")
            for i, col in enumerate(profile_cols):
                if col == 'age':
                    ages = pd.to_numeric(long_profiles['age'], errors='coerce').dropna().astype(int)
                    bins = [10, 20, 30, 40, 50, 60, 70, 100]
                    labels = ["10대", "20대", "30대", "40대", "50대", "60대", "70대+"]
                    age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
                    age_counts = age_groups.value_counts().sort_index()
                    bar_colors = [main_red] * len(age_counts)
                    if len(age_counts) > 0:
                        max_idx = age_counts.values.argmax()
                        min_idx = age_counts.values.argmin()
                        bar_colors[max_idx] = highlight  # 파랑
                        bar_colors[min_idx] = highlight_yellow  # 노랑
                    fig = go.Figure()
                    fig.add_bar(x=age_counts.index.astype(str), y=age_counts.values, marker_color=bar_colors)
                    fig.update_layout(title="age 분포 (연령대)", xaxis_title="연령대", yaxis_title="유저 수", height=200)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    counts = long_profiles[col].value_counts().head(10)
                    bar_colors = [main_red] * len(counts)
                    if len(counts) > 0:
                        max_idx = counts.values.argmax()
                        min_idx = counts.values.argmin()
                        bar_colors[max_idx] = highlight
                        bar_colors[min_idx] = highlight_yellow
                    fig = go.Figure()
                    fig.add_bar(x=counts.index.astype(str), y=counts.values, marker_color=bar_colors)
                    fig.update_layout(title=f"{col} 분포", height=200)
                    st.plotly_chart(fig, use_container_width=True)
        # 2. 구매 패턴(구매 횟수, 총 구매금액, 카테고리 선호)
        st.markdown("---")
        st.markdown("#### 구매 패턴 비교")
        pat_col1, pat_col2 = st.columns(2)
        with pat_col1:
            st.markdown("**구매주기 짧은 유저**")
            st.write(f"평균 구매 횟수: {short_orders.groupby('user_id').size().mean():.2f}")
            st.write(f"평균 총 구매금액: {short_orders.groupby('user_id')['sale_price'].sum().mean():,.0f}")
        with pat_col2:
            st.markdown("**구매주기 긴 유저**")
            st.write(f"평균 구매 횟수: {long_orders.groupby('user_id').size().mean():.2f}")
            st.write(f"평균 총 구매금액: {long_orders.groupby('user_id')['sale_price'].sum().mean():,.0f}")

with product_tab:
    st.header("제품 분석")
    # 데이터 로드 및 타입 맞추기
    if 'order_items' not in st.session_state:
        order_items = pd.read_csv('https://www.dropbox.com/scl/fi/1wus8qtn8ig32ee27nupq/order_items_preprocessed.csv?rlkey=l4a5ilw64qcpx8ukhcu79aeex&st=lux24bfy&dl=1')
        st.session_state['order_items'] = order_items
    else:
        order_items = st.session_state['order_items']

    if 'products' not in st.session_state:
        products = pd.read_csv('https://www.dropbox.com/scl/fi/nkg8tz7dn9hsmwav2vi1t/products_preprocessed.csv?rlkey=ajz1rclosfymrfw8ghmllo8vm&st=t4ccuq7k&dl=1')
        st.session_state['products'] = products
    else:
        products = st.session_state['products']

    order_items['product_id'] = order_items['product_id'].astype(str)
    products['id'] = products['id'].astype(str)

    # 1. 카테고리별 구매 전환율 (바/파이)
    order_items_cat = order_items.merge(products[['id', 'category']], left_on='product_id', right_on='id', how='left')
    category_counts = order_items_cat['category'].value_counts().sort_values(ascending=False)
    fig1 = go.Figure()
    fig1.add_bar(
        x=category_counts.index.astype(str),
        y=category_counts.values,
        marker_color='rgb(220,53,69)'
    )
    fig1.update_layout(
        title="카테고리별 구매 전환률",
        xaxis_title="카테고리",
        yaxis_title="구매 수",
        height=350
    )
    top8 = category_counts.head(8)
    etc = category_counts[8:].sum()
    pie_labels = list(top8.index) + (['기타'] if etc > 0 else [])
    pie_values = list(top8.values) + ([etc] if etc > 0 else [])
    fig1_pie = go.Figure()
    fig1_pie.add_pie(
        labels=pie_labels,
        values=pie_values,
        marker_colors=['rgb(220,53,69)', 'rgb(255,99,132)', 'rgb(255,179,186)', '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ffb3b3', '#ff4d4d'][:len(pie_labels)]
    )
    fig1_pie.update_layout(
        title="카테고리별 구매 전환률 (파이 차트, 상위 8개 + 기타)",
        width=700,
        height=350
    )

    # 2. 카테고리별 매출 합계
    category_sales = order_items_cat.groupby('category')['sale_price'].sum().sort_values(ascending=False)
    fig_category_sales = go.Figure()
    # 상위 15개 + 기타
    topN = 15
    cat_sales = category_sales.copy()
    if len(cat_sales) > topN:
        top = cat_sales.head(topN)
        etc = cat_sales.iloc[topN:].sum()
        cat_sales = pd.concat([top, pd.Series({'기타': etc})])

    cat_sales.index = [c.replace(' ', '\n') for c in cat_sales.index]

    fig_category_sales.add_bar(
        x=cat_sales.index.astype(str),
        y=cat_sales.values,
        marker_color=mid_red
    )
    fig_category_sales.update_layout(
        title="카테고리별 매출 합계 (상위 15개 + 기타)",
        xaxis_title="카테고리",
        yaxis_title="총 매출",
        height=400,
        width=1200,
        xaxis_tickangle=60,
        xaxis=dict(
            tickfont=dict(size=14, color='black')
        )
    )

    # 3. 제품별 매출 Top 10 (제품명 기준)
    order_items_name = order_items.merge(products[['id', 'name']], left_on='product_id', right_on='id', how='left')
    product_sales = order_items_name.groupby(['product_id', 'name'])['sale_price'].sum().sort_values(ascending=False).head(10).reset_index()
    fig_sales = go.Figure()
    fig_sales.add_bar(
        x=product_sales['sale_price'],
        y=product_sales['name'],
        marker_color='rgb(220,53,69)',
        orientation='h'
    )
    fig_sales.update_layout(
        title="제품별 매출 Top 10",
        xaxis_title="총 매출",
        yaxis_title="제품명",
        height=500
    )
    # 1행: 카테고리별 구매 전환율 (바/파이, 가로)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True, key="product_tab_fig1_bar")
    with col2:
        st.plotly_chart(fig1_pie, use_container_width=True, key="product_tab_fig1_pie")
    # 2행: 카테고리별 매출 합계
    st.plotly_chart(fig_category_sales, use_container_width=True, key="product_tab_fig_category_sales")
    # 3행: 제품별 매출 Top 10 (제품명 기준)
    st.plotly_chart(fig_sales, use_container_width=True, key="product_tab_fig_sales")

with region_tab:
    st.header("지역/유입 분석")

    # 2행: 국가별 유저 분포 (Plotly Choropleth)
    # 예시: users['country'] 컬럼이 있다고 가정
    country_counts = users['state'].value_counts().reset_index()
    country_counts.columns = ['country', 'user_count']

    # 로그 스케일 컬럼 추가
    country_counts['user_count_log'] = np.log1p(country_counts['user_count'])

    # 로그 값의 분위수로 색상 구간 자동 조정
    q_low = country_counts['user_count_log'].quantile(0.05)
    q_high = country_counts['user_count_log'].quantile(0.95)

    fig = px.choropleth(
        country_counts,
        locations='country',
        locationmode='country names',
        color='user_count_log',
        color_continuous_scale='YlOrRd',
        range_color=(q_low, q_high),  # 분위수로 구간 조정
        title='국가별 유저 분포 (로그 스케일)'
    )
    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        projection_type="natural earth"
    )
    fig.update_layout(
        width=1200,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=False)

    # 유입 경로별 유저 수 집계
    traffic_counts = users['traffic_source'].value_counts().sort_values(ascending=False)

    # Bar chart
    fig_traffic_bar = go.Figure()
    fig_traffic_bar.add_bar(
        x=traffic_counts.index.astype(str),
        y=traffic_counts.values,
        marker_color='rgb(220,53,69)'
    )
    fig_traffic_bar.update_layout(
        title="유입 경로별 유저 수",
        xaxis_title="유입 경로",
        yaxis_title="유저 수",
        height=350
    )

    # Pie chart
    fig_traffic_pie = go.Figure()
    fig_traffic_pie.add_pie(
        labels=traffic_counts.index.astype(str),
        values=traffic_counts.values,
        marker_colors=['rgb(220,53,69)', 'rgb(255,99,132)', 'rgb(255,179,186)', '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ffb3b3', '#ff4d4d'][:len(traffic_counts)]
    )
    fig_traffic_pie.update_layout(
        title="유입 경로별 유저 수 (파이 차트)",
        height=350
    )

    # 유입 경로별 유저 수 (Bar/Pie) 한 줄에 좌우 배치
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_traffic_bar, use_container_width=True)
    with col2:
        st.plotly_chart(fig_traffic_pie, use_container_width=True)

    # 시간대별 이벤트 발생 추이 (라인 차트)
    events = pd.read_csv('https://www.dropbox.com/scl/fi/l9bi2dfvlug1j8fy6biuo/events_preprocessed.csv?rlkey=0gxb54oymo1r7gme298jy98wv&st=gq7geawk&dl=1', parse_dates=['created_at'])
    events = events[events['user_id'].notnull()]
    events['hour'] = pd.to_datetime(events['created_at']).dt.hour
    hourly_counts = events.groupby(['hour', 'event_type']).size().unstack(fill_value=0)
    fig_hourly = go.Figure()
    for col in hourly_counts.columns:
        fig_hourly.add_trace(go.Scatter(x=hourly_counts.index, y=hourly_counts[col], mode='lines+markers', name=col))
    fig_hourly.update_layout(
        title="시간대별 이벤트 발생 추이",
        xaxis_title="시간(0~23시)",
        yaxis_title="이벤트 수",
        height=400
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

    # 이벤트별 전체 건수 (막대 차트)
    event_counts = events['event_type'].value_counts()
    fig_event_bar = go.Figure()
    fig_event_bar.add_bar(
        x=event_counts.index.astype(str),
        y=event_counts.values,
        marker_color='rgb(220,53,69)'
    )
    fig_event_bar.update_layout(
        title="이벤트별 전체 건수",
        xaxis_title="이벤트 종류",
        yaxis_title="건수",
        height=350
    )
    st.plotly_chart(fig_event_bar, use_container_width=True)
    st.write("products columns:", products.columns.tolist())