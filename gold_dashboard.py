import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
GRAMS_PER_OZ = 31.1035
EMERGENCY_EXIT_FACTOR = 0.95  # 5% below avg cost

# -------------------------------------------------
# PAGE CONFIG

# -------------------------------------------------
st.set_page_config(page_title="Gold Investment Dashboard", layout="wide")
st.title("🪙 Gold Investment Dashboard – ADCB Gold Account")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
try:
    df = pd.read_excel("gold.xlsx")
except FileNotFoundError:
    st.error("❌ gold.xlsx not found in the current folder.")
    st.stop()

df.columns = df.columns.str.strip()

required_columns = [
    "Date",
    "Quantity_oz",
    "Spent_AED",
    "Bank_Sell_oz",
    "Bank_Buy_oz",
    "Malabar"
]

for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Missing column: {col}")
        st.stop()

# -------------------------------------------------
# DATA CLEANING
# -------------------------------------------------
df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["Date"])

df["Quantity_oz"] = df["Quantity_oz"].fillna(0)
df["Spent_AED"] = df["Spent_AED"].fillna(0)

df["Bank_Sell_oz"] = df["Bank_Sell_oz"].ffill()
df["Bank_Buy_oz"] = df["Bank_Buy_oz"].ffill()
df["Malabar"] = df["Malabar"].ffill()

df.sort_values("Date", inplace=True)

# -------------------------------------------------
# CORE HOLDINGS
# -------------------------------------------------
total_oz = df["Quantity_oz"].sum()
total_grams = total_oz * GRAMS_PER_OZ
total_spent = df["Spent_AED"].sum()

avg_cost_oz = total_spent / total_oz if total_oz > 0 else 0
avg_cost_gram = avg_cost_oz / GRAMS_PER_OZ

# -------------------------------------------------
# LATEST RATES
# -------------------------------------------------
bank_sell_oz = df["Bank_Sell_oz"].iloc[-1]
bank_buy_oz = df["Bank_Buy_oz"].iloc[-1]
market_gram = df["Malabar"].iloc[-1]

bank_sell_gram = bank_sell_oz / GRAMS_PER_OZ
bank_buy_gram = bank_buy_oz / GRAMS_PER_OZ

# -------------------------------------------------
# ASSET & PROFIT (REAL EXIT)
# -------------------------------------------------
asset_if_sell = total_oz * bank_buy_oz
profit_aed = asset_if_sell - total_spent
profit_pct = (profit_aed / total_spent * 100) if total_spent > 0 else 0

profit_arrow = "⬆️" if profit_aed > 0 else "⬇️" if profit_aed < 0 else "➡️"

# -------------------------------------------------
# SPREAD, BREAK-EVEN, EMERGENCY EXIT
# -------------------------------------------------
spread_cost = (bank_sell_oz - bank_buy_oz) * total_oz
break_even_oz = avg_cost_oz
emergency_exit_oz = avg_cost_oz * EMERGENCY_EXIT_FACTOR
emergency_exit_g = avg_cost_gram * EMERGENCY_EXIT_FACTOR
# -------------------------------------------------
# SIGNAL
# -------------------------------------------------
if bank_buy_oz > avg_cost_oz:
    signal = "🟢 SELL (Profit)"
elif bank_buy_oz < emergency_exit_oz:
    signal = "🔴 LOSS ZONE"
else:
    signal = "🟡 HOLD"

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard",
    "📈 Trends",
    "📘 Logic & Definitions"
])

# -------------------------------------------------
# TAB 1 – DASHBOARD
# -------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gold Holding", f"{total_oz:.4f} oz / {total_grams:.2f} g")
    c2.metric("Total Spent", f"{total_spent:,.0f} AED")
    c3.metric("If Sell to Bank", f"{asset_if_sell:,.0f} AED")
    c4.metric("Signal", signal)

    st.divider()

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Break-Even (Avg Cost )(AED) ", f"{avg_cost_oz:.2f} oz / {avg_cost_gram:.2f} g")
    c6.metric("Emergency Exit (5% loss)", f"{emergency_exit_oz:.2f} oz / {emergency_exit_g:.2f} g")
    
    c8.metric("Spread Cost", f"{spread_cost:,.0f} AED")

    st.divider()

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Bank Sell (gram)", f"{bank_sell_oz:.2f} oz / {bank_sell_gram:.2f} g")
    c10.metric("Bank Buy (gram)", f"{bank_buy_oz:.2f} oz /  {bank_buy_gram:.2f} AED")
    c12.metric("Market 24K Bar (gram)", f"{market_gram:.2f} AED")

    st.divider()
    c13, c14, c15, c16 = st.columns(4)
    st.metric(
        "Profit / Loss",
        f"{profit_arrow} {profit_aed:,.0f} AED",
        f"{profit_pct:.2f} %"
    )

 
# -------------------------------
# WHAT-IF SIMULATOR (ONLY HERE)
# -------------------------------
st.subheader("🔮 What-If Price Simulation (ADCB Buy Price)")

# Latest Malabar price (AED/g)
latest_market_gram = df["Malabar"].iloc[-1]

scenarios = []
for pct in [-10, -5, 0, 5, 10]:
    # Future bank buy price
    future_price_oz = bank_buy_oz * (1 + pct / 100)
    future_price_g = future_price_oz / GRAMS_PER_OZ
    
    # Asset value & P/L
    future_value = future_price_oz * total_oz
    pnl = future_value - total_spent

    # Future market price in grams (2 decimal points)
    future_market_g = round(latest_market_gram * (1 + pct / 100), 2)

    scenarios.append([
        f"{pct}%",
        int(round(future_price_oz, 0)),      # Bank Buy AED/oz
        round(future_price_g, 2),            # Bank Buy AED/g
        int(round(future_value, 0)),         # Asset Value
        int(round(pnl, 0)),                  # P/L
        future_market_g                       # Market Price AED/g (2 decimals)
    ])

# Calculate break-even
break_even_price_oz = total_spent / total_oz
break_even_price_g = break_even_price_oz / GRAMS_PER_OZ
break_even_pct = (break_even_price_oz - bank_buy_oz) / bank_buy_oz * 100

break_even_market_g = round(latest_market_gram * (1 + break_even_pct / 100), 2)

scenarios.append([
    f"Break-Even ({break_even_pct:.2f}%)",
    int(round(break_even_price_oz, 0)),
    round(break_even_price_g, 2),
    int(round(total_spent, 0)),
    0,
    break_even_market_g
])

# Convert to DataFrame
sim_df = pd.DataFrame(
    scenarios,
    columns=[
        "Price Change",
        "Bank Buy (AED/oz)",
        "Bank Buy (AED/g)",
        "Asset Value (AED)",
        "P/L (AED)",
        "Market Price (AED/g)"
    ]
)

# Highlight break-even row and color P/L
def highlight_row(row):
    styles = []
    if "Break-Even" in str(row["Price Change"]):
        styles = ['background-color: #ffd966']*len(row)  # yellow
    else:
        for col in row.index:
            if col == "P/L (AED)":
                if row[col] > 0:
                    styles.append('color: green; font-weight: bold')
                elif row[col] < 0:
                    styles.append('color: red; font-weight: bold')
                else:
                    styles.append('color: orange; font-weight: bold')
            else:
                styles.append('')
    return styles

# Display in Streamlit
st.dataframe(sim_df.style.apply(highlight_row, axis=1), use_container_width=True)

## -------------------------------------------------
# TAB 2 – TRENDS (PROFESSIONAL)
# -------------------------------------------------
with tab2:

    # ===============================
    # PREP DATA (ONCE)
    # ===============================
    trend = df.copy()
    trend = trend.dropna(subset=["Date"])
    trend = trend.sort_values("Date")

    trend["Cum_Holding_oz"] = trend["Quantity_oz"].cumsum()
    trend["Cum_Spent_AED"] = trend["Spent_AED"].cumsum()

    trend = trend[trend["Cum_Holding_oz"] > 0]

    trend["ADCB_Sell_g"] = trend["Bank_Sell_oz"] / GRAMS_PER_OZ
    trend["ADCB_Buy_g"] = trend["Bank_Buy_oz"] / GRAMS_PER_OZ

    trend["Exit_Value_AED"] = trend["Cum_Holding_oz"] * trend["Bank_Buy_oz"]
    trend["Profit_AED"] = trend["Exit_Value_AED"] - trend["Cum_Spent_AED"]

    trend = trend.set_index("Date")

    

    # ======================================================
    # 📈 2. ADCB vs MARKET (GRAM)
    # ======================================================
    st.subheader("📈 ADCB vs Market (Gram)")

    gram_trend = trend[[
        "ADCB_Sell_g",
        "ADCB_Buy_g",
        "Malabar"
    ]].rename(columns={
        "ADCB_Sell_g": "ADCB Sell (g)",
        "ADCB_Buy_g": "ADCB Buy (g)",
        "Malabar": "Market 24K (g)"
    })

    st.line_chart(gram_trend, use_container_width=True)

    # ======================================================
    # 📈 3. DAILY EXIT PROFIT (WITH SIGNAL DOTS)
    # ======================================================
    st.subheader("📈 Daily Exit Profit (If Fully Sold)")

    profit_trend = trend[["Profit_AED"]].copy()

    # Identify special dates
    best_exit_date = profit_trend["Profit_AED"].idxmax()
    best_exit_value = profit_trend.loc[best_exit_date, "Profit_AED"]

    today_date = profit_trend.index.max()
    today_profit = profit_trend.loc[today_date, "Profit_AED"]

    breakeven = profit_trend.iloc[
        (profit_trend["Profit_AED"].abs()).argsort()[:1]
    ]

    st.line_chart(profit_trend, use_container_width=True)

    # ---------- SIGNAL METRICS ----------
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🔴 Best Exit Profit",
        f"{best_exit_value:,.0f} AED",
        best_exit_date.strftime("%d-%b-%Y")
    )

    c2.metric(
        "🟡 Break-even Zone",
        f"{breakeven['Profit_AED'].values[0]:,.0f} AED",
        breakeven.index[0].strftime("%d-%b-%Y")
    )

    c3.metric(
        "🟢 Current Profit",
        f"{today_profit:,.0f} AED",
        today_date.strftime("%d-%b-%Y")
    )


# -------------------------------------------------
# TAB 3 – LOGIC & DEFINITIONS (NO SIMULATOR)
# -------------------------------------------------
with tab3:
    st.write("### 📘 Definitions (Strict ADCB Logic)")
    st.write("**Average Cost / Break-Even** = Total Spent ÷ Total Ounces")
    st.write("**Profit / Loss** = (Bank Buy Price × Holding) − Total Spent")
    st.write("**Emergency Exit** = 95% of Average Cost")
    st.write("**Spread Cost** = (Bank Sell − Bank Buy) × Holding")
    st.write("**Market Price** = 24K gold bar reference (no jewellery considered)")
    st.write("**Signals** are based only on ADCB Buy price (real exit value).")

# -------------------------------------------------
# FOOTNOTE
# -------------------------------------------------
st.caption(
    "Built strictly according to ADCB gold account behavior. "
    "Market price shown only for comparison (24K bar, no making charges)."
)
# =================================================
# 🔐 ADVANCED RISK, CONFIDENCE & STRATEGY MODULE
# (Does NOT change holdings or past calculations)
# =================================================

st.divider()
st.subheader("🛡️ Risk, Confidence & Strategy (ADCB Logic)")

# -------------------------------------------------
# RISK SCORE (0–100)
# -------------------------------------------------
price_buffer_pct = (bank_buy_oz - avg_cost_oz) / avg_cost_oz if avg_cost_oz > 0 else 0
spread_penalty = (bank_sell_oz - bank_buy_oz) / bank_sell_oz

risk_score = (
    50
    + price_buffer_pct * 100
    - spread_penalty * 50
)

risk_score = max(0, min(100, risk_score))

# -------------------------------------------------
# CONFIDENCE LEVEL
# -------------------------------------------------
if risk_score >= 70:
    confidence = "🟢 HIGH CONFIDENCE"
    confidence_color = "green"
elif risk_score >= 40:
    confidence = "🟡 MEDIUM CONFIDENCE"
    confidence_color = "orange"
else:
    confidence = "🔴 LOW CONFIDENCE"
    confidence_color = "red"

# -------------------------------------------------
# PARTIAL SELL RECOMMENDATION
# -------------------------------------------------
if profit_pct > 20:
    partial_sell_pct = 40
elif profit_pct > 10:
    partial_sell_pct = 25
elif profit_pct > 5:
    partial_sell_pct = 10
else:
    partial_sell_pct = 0

partial_sell_oz = total_oz * partial_sell_pct / 100
partial_sell_grams = partial_sell_oz * GRAMS_PER_OZ
partial_sell_value = partial_sell_oz * bank_buy_oz

# -------------------------------------------------
# RE-BUY ZONE (SMART AVERAGING)
# -------------------------------------------------
rebuy_oz = avg_cost_oz * 0.97
rebuy_gram = rebuy_oz / GRAMS_PER_OZ

# -------------------------------------------------
# DISPLAY SECTION
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Risk Score",
    f"{risk_score:.0f} / 100",
)

c2.metric(
    "Confidence Meter",
    confidence
)

c3.metric(
    "Recommended Partial Sell",
    f"{partial_sell_pct} %",
    f"{partial_sell_oz:.3f} oz / {partial_sell_grams:.1f} g"
)

c4.metric(
    "Re-Buy Zone",
    f"{rebuy_oz:,.0f} AED / oz",
    f"{rebuy_gram:.2f} AED / g"
)

# -------------------------------------------------
# STRATEGY EXPLANATION
# -------------------------------------------------
with st.expander("📘 How to use this strategy"):
    st.write("""
    **Risk Score**
    - Measures how safe your position is based on *real ADCB exit price*
    - Higher score = more room to hold or wait

    **Partial Sell**
    - Locks profit without exiting fully
    - Reduces emotional pressure
    - Keeps exposure if gold keeps rising

    **Re-Buy Zone**
    - Price where adding gold improves your average cost
    - Never average UP blindly
    - This is capital-efficient buying

    **Important**
    - Signals are advisory, not automatic
    - Based strictly on ADCB buy/sell mechanics
    """)
# -------------------------------------------------
# 📊 BEST HISTORICAL SELL SCENARIO (ADCB BANK BUY)
# -------------------------------------------------

# Find historical maximum Bank Buy price
max_bank_buy_oz = df["Bank_Buy_oz"].max()

# Date when max price occurred
max_price_date = df.loc[
    df["Bank_Buy_oz"] == max_bank_buy_oz, "Date"
].iloc[0]

# Asset value if sold ALL holdings at that time
best_asset_value = total_oz * max_bank_buy_oz

# Hypothetical profit
best_profit_aed = best_asset_value - total_spent
best_profit_pct = (
    best_profit_aed / total_spent * 100
    if total_spent > 0 else 0
)

# Opportunity cost vs today
missed_profit = best_profit_aed - profit_aed

st.divider()
st.subheader("📈 Best Possible Historical Exit (ADCB Logic)")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Max Bank Buy Price",
    f"{max_bank_buy_oz:,.0f} AED / oz",
    max_price_date.strftime("%d %b %Y")
)

c2.metric(
    "Asset Value If Sold Then",
    f"{best_asset_value:,.0f} AED"
)

c3.metric(
    "Profit If Sold at Peak",
    f"{best_profit_aed:,.0f} AED",
    f"{best_profit_pct:.2f} %"
)

c4.metric(
    "Missed Opportunity vs Today",
    f"{missed_profit:,.0f} AED"
)

# =================================================
# 🔐 ADVANCED RISK, CONFIDENCE & STRATEGY MODULE
# =================================================
# =================================================
# 📊 PROFESSIONAL PERFORMANCE KPIs (CORRECTED)
# =================================================
st.divider()
st.subheader("📊 Professional Performance KPIs (Correct Holdings Logic)")

# -------------------------------------------------
# 1️⃣ DAILY HOLDINGS (CUMULATIVE, ADCB-STYLE)
# -------------------------------------------------
df["Holding_oz"] = df["Quantity_oz"].cumsum()

# -------------------------------------------------
# 2️⃣ DAILY EXIT VALUE (REALISTIC)
# -------------------------------------------------
df["Exit_Value"] = df["Holding_oz"] * df["Bank_Buy_oz"]

# -------------------------------------------------
# 3️⃣ BEST POSSIBLE EXIT
# -------------------------------------------------
best_exit_row = df.loc[df["Exit_Value"].idxmax()]

best_exit_value = best_exit_row["Exit_Value"]
best_exit_price = best_exit_row["Bank_Buy_oz"]
best_exit_date = best_exit_row["Date"]

# -------------------------------------------------
# 4️⃣ CURRENT REAL VALUE (REALIZED + UNREALIZED)
# -------------------------------------------------
current_unrealized = total_oz * bank_buy_oz
current_total_value = current_unrealized + realized_cash if "realized_cash" in locals() else current_unrealized

# -------------------------------------------------
# 5️⃣ MISSED UPSIDE (CORRECT)
# -------------------------------------------------
missed_upside = best_exit_value - current_total_value

# -------------------------------------------------
# 6️⃣ MAX PROFIT POSSIBLE (AT THAT TIME)
# -------------------------------------------------
# Cost basis at that date
spent_until_best_exit = df.loc[df["Date"] <= best_exit_date, "Spent_AED"].sum()

max_profit_possible = best_exit_value - spent_until_best_exit

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
k1, k2, k3 = st.columns(3)

k1.metric(
    "Best Possible Exit Value",
    f"{best_exit_value:,.0f} AED",
    best_exit_date.strftime("%d %b %Y")
)

k2.metric(
    "Max Profit Possible (Then)",
    f"{max_profit_possible:,.0f} AED"
)

k3.metric(
    "Missed Upside vs Best Exit",
    f"{missed_upside:,.0f} AED"
)

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------
with st.expander("📘 What does this mean?"):
    st.write("""
    **Best Possible Exit**
    - The single best day where selling *everything you held that day*
      at ADCB Bank Buy price would have maximized value.

    **Missed Upside**
    - How much more money you *could have had*
      if you exited perfectly,
      compared to what you actually have today
      (cash already received + current gold value).

    ✔ Uses real holdings on each day  
    ✔ Fully supports partial buys and partial sells  
    ✔ No inflation or hindsight bias  
    """)
