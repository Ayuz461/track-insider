import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf
import io
from datetime import timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Configure Streamlit app
st.set_page_config(page_title="Insider Trading Tracker", layout="wide")
st.title("📊 Insider Trading Tracker & Visualizer")

@st.cache_data
def load_data():
    url = "http://openinsider.com/screener?type=&amount=10000&owner=only&sortCol=0&sortDir=desc"
    try:
        tables = pd.read_html(url)
        df = tables[11]
        df.columns = df.columns.map(str).str.replace('\xa0', ' ', regex=False).str.strip()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load and clean the dataset
df = load_data()
df['Price'] = df['Price'].astype(str).str.extract(r'\$?([\d,.]+)', expand=False)
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")

if df.empty:
    st.stop()

# Sidebar filtering options
st.sidebar.title("📊 Filter Options")

start_date = st.sidebar.date_input("Start Date", value=df["Trade Date"].min().date())
end_date = st.sidebar.date_input("End Date", value=df["Trade Date"].max().date())

filtered_df = df[(df["Trade Date"].dt.date >= start_date) & (df["Trade Date"].dt.date <= end_date)]

trade_type = st.sidebar.selectbox("Select Trade Type", ["All", "Buy (Purchase)", "Sell"])
if trade_type == "Buy (Purchase)":
    filtered_df = filtered_df[filtered_df['Trade Type'].str.contains("Purchase", na=False)]
elif trade_type == "Sell":
    filtered_df = filtered_df[filtered_df['Trade Type'].str.contains("Sell", na=False)]

all_tickers = sorted(filtered_df["Ticker"].dropna().unique())
selected_tickers = st.sidebar.multiselect("Filter by Ticker(s)", options=all_tickers, default=all_tickers[:10])
if selected_tickers:
    filtered_df = filtered_df[filtered_df["Ticker"].isin(selected_tickers)]

all_insiders = sorted(filtered_df["Insider Name"].dropna().unique())
selected_insiders = st.sidebar.multiselect("Filter by Insider(s)", options=all_insiders)
if selected_insiders:
    filtered_df = filtered_df[filtered_df["Insider Name"].isin(selected_insiders)]

search_term = st.sidebar.text_input("Search by Ticker or Company Name", value="")
if search_term:
    search_term = search_term.lower()
    filtered_df = filtered_df[
        filtered_df["Ticker"].str.lower().str.contains(search_term) |
        filtered_df["Company Name"].str.lower().str.contains(search_term)
    ]

st.sidebar.markdown(f"Showing **{len(filtered_df)}** filtered trades")

# Define tabs
tabs = st.tabs([
    "📋 Overview", "📉 Price Impact", "📊 Charts", "🏢 Clustering", "💾 Export",
    "🧾 Trade Table", "🚨 Anomaly Detection (ML)", "📈 Anomaly Timeline", "⚠️ Risk Scoring"
])

# Tab 0 - Overview
with tabs[0]:
    st.subheader("📊 Insider Summary Stats")
    total_trades = len(filtered_df)
    total_insiders = filtered_df['Insider Name'].nunique()
    total_companies = filtered_df['Ticker'].nunique()

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Total Trades", f"{total_trades:,}")
    col_kpi2.metric("Unique Insiders", f"{total_insiders:,}")
    col_kpi3.metric("Unique Companies", f"{total_companies:,}")

    total_buy_volume = filtered_df.loc[filtered_df['Trade Type'].str.contains('Purchase', na=False), 'Qty'].sum()
    total_sell_volume = filtered_df.loc[filtered_df['Trade Type'].str.contains('Sale', na=False), 'Qty'].sum()
    avg_buy_price = filtered_df.loc[filtered_df['Trade Type'].str.contains('Purchase', na=False), 'Price'].mean()
    avg_sell_price = filtered_df.loc[filtered_df['Trade Type'].str.contains('Sale', na=False), 'Price'].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Buy Volume", f"{total_buy_volume:,.0f}")
        st.metric("Avg Buy Price", f"${avg_buy_price:.2f}")
    with col2:
        st.metric("Total Sell Volume", f"{total_sell_volume:,.0f}")
        st.metric("Avg Sell Price", f"${avg_sell_price:.2f}")

    st.markdown("**👤 Top 5 Insiders by Trade Count:**")
    top_insiders = filtered_df['Insider Name'].value_counts().head(5)
    st.bar_chart(top_insiders)

# Tab 1 - Price Impact
with tabs[1]:
    st.subheader("📉 Price Change Impact Around Trade Date")

    def get_price_change(ticker, trade_date):
        try:
            trade_date = pd.to_datetime(trade_date)
            start_date = trade_date - timedelta(days=10)
            end_date = trade_date + timedelta(days=10)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if data.empty or 'Close' not in data.columns:
                return None, None, None
            close_prices = data['Close'].copy().ffill().bfill()
            prev_days = close_prices.index[close_prices.index < trade_date]
            next_days = close_prices.index[close_prices.index > trade_date]
            if len(prev_days) == 0 or len(next_days) == 0:
                return None, None, None

            price_before = float(close_prices.loc[prev_days[-1]].iloc[0])
            price_after = float(close_prices.loc[next_days[0]].iloc[0])
            change_percent = ((price_after - price_before) / price_before) * 100
            return round(price_before, 2), round(price_after, 2), round(change_percent, 2)
        except Exception:
            return None, None, None

    sample_df = filtered_df.head(10)
    price_change_data = []
    with st.spinner("Fetching price changes..."):
        for _, row in sample_df.iterrows():
            ticker = row["Ticker"]
            trade_date = row["Trade Date"]
            before, after, change = get_price_change(ticker, trade_date)
            price_change_data.append({
                "Ticker": ticker,
                "Trade Date": trade_date.date(),
                "Price Before": before,
                "Price After": after,
                "% Change (±)": f"{change}%" if change is not None else "N/A"
            })
    st.dataframe(pd.DataFrame(price_change_data))

# Tab 2 - Charts
with tabs[2]:
    st.subheader("📊 Visual Insights")
    top_tickers = filtered_df["Ticker"].value_counts().head(10).reset_index()
    top_tickers.columns = ["Ticker", "Trade Count"]
    fig1 = px.bar(top_tickers, x="Ticker", y="Trade Count", color="Trade Count",
                  title="Top 10 Tickers by Trade Count", color_continuous_scale="Blues")
    st.plotly_chart(fig1, use_container_width=True)

    buy_count = filtered_df["Trade Type"].str.contains("Purchase", na=False).sum()
    sell_count = filtered_df["Trade Type"].str.contains("Sale", na=False).sum()
    pie_data = pd.DataFrame({"Type": ["Buy", "Sell"], "Count": [buy_count, sell_count]})
    fig2 = px.pie(pie_data, names="Type", values="Count", color="Type",
                  color_discrete_map={"Buy": "green", "Sell": "red"},
                  title="Buy vs Sell Trade Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3 - Clustering
with tabs[3]:
    st.subheader("🏢 Company-Level Trade Clustering")
    cluster_df = filtered_df.groupby("Ticker")["Insider Name"].nunique().reset_index()
    cluster_df.columns = ["Ticker", "Unique Insiders"]
    top_clusters = cluster_df.sort_values("Unique Insiders", ascending=False).head(10)
    fig3 = px.bar(top_clusters, x="Ticker", y="Unique Insiders", color="Unique Insiders",
                  color_continuous_scale="Viridis", title="Top 10 Companies by Insider Clustering")
    st.plotly_chart(fig3, use_container_width=True)

    vol_df = filtered_df.groupby("Ticker")["Qty"].sum().reset_index().sort_values("Qty", ascending=False).head(10)
    vol_df.columns = ["Ticker", "Total Volume"]
    fig4 = px.bar(vol_df, x="Ticker", y="Total Volume", color="Total Volume",
                  color_continuous_scale="Blues", title="Top 10 Companies by Insider Trade Volume")
    st.plotly_chart(fig4, use_container_width=True)

# Tab 4 - Export
with tabs[4]:
    st.subheader("💾 Export Filtered Data")

    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='FilteredData')
        return output.getvalue()

    csv_data = convert_df_to_csv(filtered_df)
    excel_data = convert_df_to_excel(filtered_df)
    st.download_button("📥 Download CSV", data=csv_data, file_name='insider_trades_filtered.csv', mime='text/csv')
    st.download_button("📥 Download Excel", data=excel_data, file_name='insider_trades_filtered.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Tab 5 - Table
with tabs[5]:
    st.subheader("🧾 Insider Trade Details")
    rows_per_page = 50
    total_rows = len(filtered_df)
    page = st.number_input("Page", min_value=1, max_value=(total_rows // rows_per_page) + 1, value=1, step=1)
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    st.dataframe(filtered_df.iloc[start_idx:end_idx])
# Tab 6 - Anomaly Detection (ML)
with tabs[6]:
    st.subheader("🚨 Anomaly Detection using Isolation Forest")
    st.markdown("""
    This tool identifies trades that deviate significantly from the norm based on trade price, quantity, volume, recency, and type.
    """)

    model_df = filtered_df.copy()
    model_df = model_df.dropna(subset=["Price", "Qty", "Trade Date", "Trade Type"])
    model_df["Volume"] = model_df["Price"] * model_df["Qty"]
    model_df["Days Ago"] = (pd.Timestamp.today() - model_df["Trade Date"]).dt.days

    le = LabelEncoder()
    model_df["Trade Type Encoded"] = le.fit_transform(model_df["Trade Type"].astype(str))

    features = ["Price", "Qty", "Volume", "Days Ago", "Trade Type Encoded"]
    X = model_df[features]

    if X.empty:
        st.warning("Not enough data for anomaly detection.")
    else:
        contamination_rate = st.slider("Contamination Rate (Outlier Ratio)", 0.01, 0.2, 0.05, 0.01)
        model = IsolationForest(contamination=contamination_rate, random_state=42)
        model_df["Anomaly"] = model.fit_predict(X)
        model_df["Anomaly Score"] = -model.decision_function(X)

        anomalies = model_df[model_df["Anomaly"] == -1].copy()

        def describe_anomaly(row):
            diffs = {col: abs(row[col] - X[col].mean()) / (X[col].std() + 1e-9) for col in features[:-1]}
            key_feature = max(diffs, key=diffs.get)
            direction = "high" if row[key_feature] > X[key_feature].mean() else "low"
            return f"{key_feature} unusually {direction}"

        anomalies["Reason"] = anomalies.apply(lambda r: describe_anomaly(r), axis=1)

        st.success(f"Detected {len(anomalies)} anomalous trades")
        st.dataframe(anomalies[["Ticker", "Insider Name", "Price", "Qty", "Trade Date", "Trade Type", "Anomaly Score", "Reason"]].sort_values("Anomaly Score", ascending=False))

        fig = px.scatter(model_df, x="Volume", y="Price", color=model_df["Anomaly"].map({1: "Normal", -1: "Anomaly"}),
                         hover_data=["Ticker", "Insider Name", "Qty"],
                         title="Trade Volume vs Price with Anomaly Highlight")
        st.plotly_chart(fig, use_container_width=True)

# Tab 7 - Anomaly Timeline
with tabs[7]:
    st.subheader("📈 Timeline of Anomalies")
    if not anomalies.empty:
        anomaly_trend = anomalies.groupby("Trade Date").size().reset_index(name="Anomaly Count")
        fig = px.line(anomaly_trend, x="Trade Date", y="Anomaly Count", markers=True, title="Anomalies Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies detected for timeline visualization.")

# Tab 8 - Risk Scoring
with tabs[8]:
    st.subheader("⚠️ Insider Trade Risk Scoring")
    st.markdown("""
    This score reflects how significant and potentially suspicious a trade is, based on:
    - **Volume** (Price × Quantity)
    - **Recency** (More recent trades are more significant)
    - **Trade Type** (Sales are generally considered more risky than purchases)

    **Scoring Weights:**
    - Volume Score: 40%
    - Recency Score: 30%
    - Trade Type Score: 30%

    **Trade Type Encoding:**
    - Sale = 2
    - Purchase = 1
    - Others = 0

    **Scoring Formula:**
    `Total Score = 0.4 × Volume Score + 0.3 × Recency Score + 0.3 × Trade Type Score`
    """)

    risk_df = filtered_df.copy().dropna(subset=["Price", "Qty", "Trade Date", "Trade Type"])
    risk_df["Volume"] = risk_df["Price"] * risk_df["Qty"]
    risk_df["Days Ago"] = (pd.Timestamp.today() - risk_df["Trade Date"]).dt.days
    risk_df["Trade Type Score"] = risk_df["Trade Type"].apply(lambda t: 2 if "Sale" in str(t) else 1 if "Purchase" in str(t) else 0)

    def normalize(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-9)

    risk_df["Volume Score"] = normalize(risk_df["Volume"])
    risk_df["Recency Score"] = 1 - normalize(risk_df["Days Ago"])
    risk_df["Trade Type Score Norm"] = normalize(risk_df["Trade Type Score"])
    risk_df["Total Score"] = 0.4 * risk_df["Volume Score"] + 0.3 * risk_df["Recency Score"] + 0.3 * risk_df["Trade Type Score Norm"]

    def assign_risk(score):
        if score >= 0.8:
            return "🔴 High Risk"
        elif score >= 0.5:
            return "🟠 Moderate Risk"
        return "🟢 Low Risk"

    risk_df["Risk Level"] = risk_df["Total Score"].apply(assign_risk)
    st.dataframe(risk_df[["Ticker", "Insider Name", "Trade Date", "Price", "Qty", "Volume", "Days Ago", "Total Score", "Risk Level"]].sort_values("Total Score", ascending=False).head(20))

    fig = px.scatter(risk_df, x="Volume", y="Days Ago", color="Total Score",
                     hover_data=["Ticker", "Insider Name", "Trade Type"],
                     color_continuous_scale="Reds",
                     title="Trade Risk by Volume and Recency")
    st.plotly_chart(fig, use_container_width=True)
