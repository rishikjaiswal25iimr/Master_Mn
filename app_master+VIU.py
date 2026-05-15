"""
UNIFIED MANGANESE FORECASTING DASHBOARD
=================================================
Handles Mn Briquette (97%), LC FeMn (80%), and MC FeMn (70%).
NEW: Fixed Penalty Application (No Eff Mn division) & 3-Year Future TCO Chart.
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import linprog

st.set_page_config(page_title="Manganese Intelligence", page_icon="📈", layout="wide")

C_ACTUAL   = "#FF9800"
C_HYBRID   = "#2196F3"
C_FUTURE   = "#4CAF50"
C_MARKET   = "#9C27B0"
C_REG1     = "rgba(244, 67, 54, 0.15)"
C_GRID     = "#EEEEEE"
C_TEXT     = "#333333"
C_CI       = "rgba(76, 175, 80, 0.12)"

HORIZON_OPTIONS = {"4 weeks (1 month)": 4, "12 weeks (3 months)": 12, "26 weeks (6 months)": 26, "52 weeks (1 year)": 52, "104 weeks (2 years)": 104, "156 weeks (3 years)": 156}

def _layout(title: str, y_title: str = "Price", height: int = 460) -> dict:
    return dict(template="plotly_white", paper_bgcolor="white", plot_bgcolor="#FAFAFA", font=dict(family="sans-serif", size=12, color=C_TEXT), title=dict(text=title, font=dict(size=16, color="#111"), x=0.01), legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#CCC", borderwidth=1), xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False), yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title), hovermode="x unified", height=height, margin=dict(l=55, r=20, t=55, b=40))

def build_regime_shapes(dates: pd.DatetimeIndex, probs: np.ndarray, threshold: float = 0.5) -> list:
    labels = (probs > threshold).astype(int)
    shapes, in_block, t0 = [], False, None
    for d, lbl in zip(dates, labels):
        if lbl == 1 and not in_block:
            in_block, t0 = True, d
        elif lbl == 0 and in_block:
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=str(t0), x1=str(d), y0=0, y1=1, fillcolor=C_REG1, line_width=0, layer="below"))
            in_block = False
    if in_block: shapes.append(dict(type="rect", xref="x", yref="paper", x0=str(t0), x1=str(dates[-1]), y0=0, y1=1, fillcolor=C_REG1, line_width=0, layer="below"))
    return shapes

with st.sidebar:
    st.markdown("## ⚙️ Target Selection")
    alloy_choice = st.radio("Select Manganese Product:", ["Mn Briquette (97%)", "LC FeMn (80%)", "MC FeMn (70%)"])
    
    if "Briquette" in alloy_choice: alloy_code = "Briquette"
    elif "LC" in alloy_choice: alloy_code = "LC"
    else: alloy_code = "MC"
    
    output_dir = f"./outputs_{alloy_code}"
    
    st.divider()
    display_window = st.slider("History to show (weeks)", 52, 520, 260, step=26)
    horizon_label  = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()), index=2)
    price_mode     = st.radio("Display prices as", ["Real price (Rs/Kg)", "Index value"], index=0)
    show_regime    = st.checkbox("Show regime shading", value=True)
    show_market    = st.checkbox("Show Market Price", value=True)
    show_dual      = st.checkbox("Show Dual-Axis Chart", value=False)

try:
    hist = pd.read_csv(f"{output_dir}/historical_predictions.csv", index_col=0, parse_dates=True)
    future = pd.read_csv(f"{output_dir}/future_forecast.csv", index_col=0, parse_dates=True).iloc[:HORIZON_OPTIONS[horizon_label]]
    fi = pd.read_csv(f"{output_dir}/feature_importance.csv").sort_values("importance", ascending=False).head(15)
    with open(f"{output_dir}/model_metadata.json") as f: meta = json.load(f)
except FileNotFoundError:
    st.error(f"⛔ Data for {alloy_code} not found. Please run the pipeline using `python pipeline_master.py --alloy {alloy_code}` first.")
    st.stop()

st.markdown("## 📈 Mn Price Forecasting Engine")

pred_col = "hybrid_prediction"
mape = float(np.mean(np.abs((hist["actual"] - hist[pred_col]) / (hist["actual"] + 1e-9))) * 100)

c1, c2, c3, c4, c5, c6 = st.columns(6)
last_idx = float(hist["actual"].iloc[-1])
last_real = float(hist["real_price"].iloc[-1])
nxt_real = float(future["real_price"].iloc[0])
end_real = float(future["real_price"].iloc[-1])
pct_chg = ((nxt_real - last_real) / last_real) * 100 if last_real else 0

c1.metric("Last Index", f"{last_idx:.2f}")
c2.metric("Last Actual (Rs/Kg)", f"₹{last_real:.2f}")
c3.metric("Next-Wk Forecast", f"₹{nxt_real:.2f}", delta=f"{pct_chg:+.1f}%")
c4.metric("End Forecast", f"₹{end_real:.2f}", delta=f"{len(future)} wks ahead", delta_color="off")
c5.metric("In-Sample MAPE", f"{mape:.2f}%")
c6.metric("Scaling Factor", f"{meta.get('scaling_factor', 0):.4f}", help=f"Anchored mathematically to {meta.get('anchor_date')} = ₹{meta.get('anchor_price', 0):.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 Price Forecast", "📊 Market Comparison", "🔀 Regime & Drivers", "⚖️ VIU & TCO Optimizer", "🧠 Substitution Solver"])

# ══ TAB 1: PRICE FORECAST ════════════════════════════════════════════════════
with tab1:
    h = hist[hist.index >= hist.index[-1] - pd.DateOffset(weeks=display_window)].copy()
    if price_mode == "Real price (Rs/Kg)":
        actual_vals, y_title, future_col, fmt = h["real_price"], "Price (Rs/Kg)", "real_price", "₹%{y:.2f}"
    else:
        actual_vals, y_title, future_col, fmt = h["actual"], "Price Index", "predicted_index", "%{y:.2f}"

    fig1 = go.Figure()
    if show_regime and "regime_probability" in h.columns:
        for s in build_regime_shapes(h.index, h["regime_probability"].values): fig1.add_shape(**s)

    full_x = list(h.index) + list(future.index)
    N, M = len(h), len(future)

    fig1.add_trace(go.Scatter(x=full_x, y=list(actual_vals) + [None]*M, name="Index Price", line=dict(color=C_HYBRID, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index Price: " + fmt + "<extra></extra>"))
    if show_market and "market_price" in h.columns:
        fig1.add_trace(go.Scatter(x=full_x, y=list(h["market_price"]) + [None]*M, name="Market Price", line=dict(color=C_MARKET, width=2.5), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Market Price: " + fmt + "<extra></extra>"))

    conn_val = actual_vals.iloc[-1]
    fp = future[future_col]
    fut_y = [None]*(N-1) + [conn_val] + list(fp.values)
    
    idx_arr = np.arange(1, M + 1)
    sigma = np.std(fp.values) * 0.015 * idx_arr
    ci_up, ci_dn = list(fp.values + 1.96 * sigma), list(fp.values - 1.96 * sigma)
    ci_x = [h.index[-1]] + list(future.index)
    fig1.add_trace(go.Scatter(x=ci_x + ci_x[::-1], y=[conn_val] + ci_up + ([conn_val] + ci_dn)[::-1], fill="toself", fillcolor=C_CI, line=dict(width=0), name="95% CI", showlegend=True, hoverinfo="skip"))

    custom_data = np.stack(([None]*(N-1) + [conn_val] + ci_up, [None]*(N-1) + [conn_val] + ci_dn), axis=-1)
    htemplate = ("<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Forecast Price: " + fmt + "<br>Max (95% CI): " + fmt.replace("%{y", "%{customdata[0]") + "<br>Min (95% CI): " + fmt.replace("%{y", "%{customdata[1]") + "<extra></extra>")
    fig1.add_trace(go.Scatter(x=full_x, y=fut_y, name="Future Forecast Price", line=dict(color=C_FUTURE, width=3), customdata=custom_data, hovertemplate=htemplate))

    fig1.update_layout(**_layout(f"{alloy_choice} Price Trajectory", y_title, 470))
    st.plotly_chart(fig1, use_container_width=True)

    if show_dual:
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        h_idx_vals = h["hybrid_prediction"]
        
        fig2.add_trace(go.Scatter(x=full_x, y=list(h_idx_vals) + [None]*M, name="Index", line=dict(color=C_HYBRID, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index: %{y:.2f}<extra></extra>"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=full_x, y=list(h["real_price"]) + [None]*M, name="Index Price", line=dict(color=C_ACTUAL, width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index Price: ₹%{y:.2f}<extra></extra>"), secondary_y=True)

        fut_idx_y = [None]*(N-1) + [h_idx_vals.iloc[-1]] + list(future["predicted_index"])
        fig2.add_trace(go.Scatter(x=full_x, y=fut_idx_y, name="Future Index", line=dict(color="#64B5F6", width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Index: %{y:.2f}<extra></extra>"), secondary_y=False)
        
        fut_rp_y = [None]*(N-1) + [h["real_price"].iloc[-1]] + list(future["real_price"])
        fig2.add_trace(go.Scatter(x=full_x, y=fut_rp_y, name="Future Index Price", line=dict(color="#FF5722", width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Index Price: ₹%{y:.2f}<extra></extra>"), secondary_y=True)

        fig2.update_layout(**_layout("Index vs Price — Dual Axis Comparison", "Price Index", 400))
        fig2.update_yaxes(title_text="Price Index (Solid Lines)", secondary_y=False)
        fig2.update_yaxes(title_text="Price Rs/Kg (Dashed Lines)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig2, use_container_width=True)

# ══ TAB 2: MARKET COMPARISON ═════════════════════════════════════════════════
with tab2:
    if "market_price" in hist.columns:
        common = hist[["real_price", "market_price"]].dropna()
        if not common.empty:
            fig_mkt = go.Figure()
            fig_mkt.add_trace(go.Scatter(x=common.index, y=common["market_price"], name="Market Price", mode="lines", line=dict(color=C_MARKET, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Market Price: ₹%{y:.2f}<extra></extra>"))
            fig_mkt.add_trace(go.Scatter(x=common.index, y=common["real_price"], name="Index Price", mode="lines", line=dict(color=C_HYBRID, width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index Price: ₹%{y:.2f}<extra></extra>"))
            fig_mkt.update_layout(**_layout("Market Comparison: Real vs Predicted", y_title="Rs/Kg", height=380))
            st.plotly_chart(fig_mkt, use_container_width=True)
            err = common["real_price"] - common["market_price"]
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("RMSE vs Market", f"₹{float(np.sqrt((err**2).mean())):.2f}")
            ec2.metric("MAE vs Market", f"₹{float(err.abs().mean()):.2f}")
            ec3.metric("MAPE vs Market", f"{float((err.abs() / common['market_price']).mean() * 100):.2f}%")
        else: st.info("📂 Market column exists, but dates did not match.")
    else: st.info("📂 No market price data found. The pipeline skipped it.")

# ══ TAB 3: REGIME & DRIVERS ══════════════════════════════════════════════════
with tab3:
    st.markdown("### 🌍 Underlying Macro Drivers Over Time")
    st.caption("Tracks the raw online data inputs. Normalized to start at 100 on the left side to compare relative growth.")
    
    driver_list = ["simn", "mn_ore", "chn_electricity", "met_coal", "dry_bulk_freight", "steel_etf", "usd_inr", "pick"]
    available_drivers = [d for d in driver_list if d in hist.columns]
    
    if available_drivers:
        fig_d = go.Figure()
        plot_h = hist[hist.index >= hist.index[-1] - pd.DateOffset(weeks=display_window)]
        for d_col in available_drivers:
            first_valid = plot_h[d_col].dropna().iloc[0] if not plot_h[d_col].dropna().empty else 1.0
            fig_d.add_trace(go.Scatter(x=plot_h.index, y=(plot_h[d_col] / (first_valid + 1e-9)) * 100, mode='lines', name=d_col.replace("_", " ").title()))
        fig_d.update_layout(**_layout("Relative Movement of Key Drivers (Indexed to 100)", "Index Value", 400))
        st.plotly_chart(fig_d, use_container_width=True)
        
    st.divider()

    if "regime_probability" in hist.columns:
        st.markdown("### Market Regime State")
        fig_r = go.Figure(go.Scatter(x=h.index, y=h["regime_probability"], name="P(Supply Squeeze)", fill="tozeroy", fillcolor="rgba(244, 67, 54, 0.2)", line=dict(color="#D32F2F")))
        fig_r.update_layout(**_layout("P(Supply Squeeze / High Volatility)", "Probability", 250))
        st.plotly_chart(fig_r, use_container_width=True)

    fc1, fc2 = st.columns(2)
    fig_bar = go.Figure(go.Bar(x=fi["importance"][::-1], y=fi["feature"][::-1], orientation="h", marker=dict(color="#3F51B5", opacity=0.85)))
    fig_bar.update_layout(**_layout("Top Driver Importance", "Score", 400))
    fc1.plotly_chart(fig_bar, use_container_width=True)

    top_d = fi.head(7).copy()
    if fi.iloc[7:]["importance"].sum() > 0: top_d = pd.concat([top_d, pd.DataFrame([{"feature": "Other Variables", "importance": fi.iloc[7:]["importance"].sum()}])])
    fig_pie = go.Figure(data=[go.Pie(labels=top_d["feature"], values=(top_d["importance"] / top_d["importance"].sum()) * 100, hole=0.5)])
    fig_pie.update_layout(title=dict(text=f"% Dependence of {alloy_code} on Drivers", font=dict(size=16), x=0.5), template="plotly_white", height=400)
    fc2.plotly_chart(fig_pie, use_container_width=True)


# ══ DATA INITIALIZATION FOR TAB 4 & 5 ════════════════════════════════════════
def get_latest_price(folder):
    try:
        return float(pd.read_csv(f"{folder}/historical_predictions.csv")["real_price"].iloc[-1]) * 1000
    except: return None

def get_future_prices(folder):
    try:
        df = pd.read_csv(f"{folder}/future_forecast.csv", index_col=0, parse_dates=True)
        return df["real_price"] * 1000
    except: return None

p_mc = get_latest_price("./outputs_MC") or 108500
p_lc = get_latest_price("./outputs_LC") or 140000
p_bq = get_latest_price("./outputs_Briquette") or 240000

# Physics & Valuation Constants (From Excel Solver Base)
HEAT_SIZE = 190.0
POWER_RATE = 6.5
ELECTRODE_RATE = 240.0
THROUGHPUT_MARGIN = 2800.0
REBLOW_COST = 15000.0
STEEL_VALUE = 60000.0
RH_COST = 2500.0

def compute_operational_penalty(thermal, cycle, rec_diff, reblow, clean, yield_loss, rh, eff_mn, price):
    # 1 MT of Alloy supports 200T of steel. Heat size is 190T.
    scale = 200.0 / HEAT_SIZE
    
    power_kwh_per_heat = (thermal * 0.75 * HEAT_SIZE) / 3.6 / 0.78
    power = power_kwh_per_heat * POWER_RATE * scale
    electrode = power_kwh_per_heat * 0.0028 * ELECTRODE_RATE * scale
    throughput = (cycle / 53.0) * HEAT_SIZE * 0.30 * THROUGHPUT_MARGIN * scale
    rec_ben = ((rec_diff / 100.0) / eff_mn) * price * 0.50
    reblow_ben = (reblow / 100.0) * REBLOW_COST * scale
    rh_ben = (rh / 100.0) * RH_COST * scale
    clean_ben = STEEL_VALUE * (clean / 100.0) * 200.0 * 0.40
    yield_ben = STEEL_VALUE * (yield_loss / 100.0) * 200.0 * 0.50
    
    return power + electrode + throughput + rec_ben + reblow_ben + clean_ben + yield_ben + rh_ben


# ══ TAB 4: ADVANCED VIU & TCO DASHBOARD ═════════════════════════════════════
with tab4:
    st.markdown("### ⚖️ Total Cost of Ownership (TCO) & Procurement Optimizer")
    
    active_alloys = st.multiselect("Select Alloys to Compare", ["MC FeMn", "LC FeMn", "Mn Briquette"], default=["MC FeMn", "LC FeMn", "Mn Briquette"])
    
    st.markdown("#### 1. Current Market Prices (₹/MT)")
    c1, c2, c3, c4 = st.columns(4)
    mc_price = c1.number_input("MC FeMn Price", value=int(p_mc), step=1000)
    lc_price = c2.number_input("LC FeMn Price", value=int(p_lc), step=1000)
    bq_price = c3.number_input("Mn Briquette Price", value=int(p_bq), step=1000)
    scrap_price = c4.number_input("Scrap Value (Fe Credit)", value=35000, step=1000)

    with st.expander("⚙️ Advanced Operational Parameters & Penalty Adjustments", expanded=False):
        ec1, ec2, ec3 = st.columns(3)
        
        with ec1:
            st.markdown("**MC FeMn Constraints**")
            mc_mn = st.slider("MC Mn %", 60.0, 80.0, 70.0) / 100
            mc_rec = st.slider("MC Recovery %", 70.0, 95.0, 85.0) / 100
            mc_fe = st.slider("MC Fe %", 10.0, 30.0, 20.0) / 100
            mc_carbon = st.number_input("MC Carbon %", value=1.50, step=0.1)
            mc_eff = mc_mn * mc_rec
            
            st.markdown("*VIU Deltas (vs Briquette)*")
            mc_thermal = st.slider("Net Thermal Pickup ΔT (°C)", 0.0, 6.0, 4.0, key="m_t", step=0.5)
            mc_cycle = st.slider("Cycle Time Saved (mins)", 0.0, 5.0, 1.14, key="m_c", step=0.01)
            mc_rec_diff = st.slider("Recovery Buffer Δ (%)", 0.0, 5.0, 2.0, key="m_r", step=0.1)
            mc_reblow = st.slider("Reblow Risk Δ (%)", 0.0, 5.0, 1.0, key="m_rb", step=0.1)
            mc_clean = st.slider("Cleanliness Rejection Δ (%)", 0.000, 0.100, 0.020, format="%.3f", key="m_cl")
            mc_yield = st.slider("Yield Loss Δ (%)", 0.000, 0.100, 0.030, format="%.3f", key="m_y")
            mc_rh = st.slider("RH / Carbon Correction Δ (%)", 0.0, 20.0, 10.0, key="m_rh", step=0.5)
            
            mc_penalty = compute_operational_penalty(mc_thermal, mc_cycle, mc_rec_diff, mc_reblow, mc_clean, mc_yield, mc_rh, mc_eff, mc_price)
            st.info(f"**Total Operational Penalty: ₹{mc_penalty:,.0f} / MT**")

        with ec2:
            st.markdown("**LC FeMn Constraints**")
            lc_mn = st.slider("LC Mn %", 70.0, 90.0, 80.0) / 100
            lc_rec = st.slider("LC Recovery %", 75.0, 98.0, 90.0) / 100
            lc_fe = st.slider("LC Fe %", 5.0, 25.0, 15.0) / 100
            lc_carbon = st.number_input("LC Carbon %", value=0.50, step=0.01)
            lc_eff = lc_mn * lc_rec
            
            st.markdown("*VIU Deltas (vs Briquette)*")
            lc_thermal = st.slider("Net Thermal Pickup ΔT (°C)", 0.0, 6.0, 1.5, key="l_t", step=0.5)
            lc_cycle = st.slider("Cycle Time Saved (mins)", 0.0, 5.0, 0.43, key="l_c", step=0.01)
            lc_rec_diff = st.slider("Recovery Buffer Δ (%)", 0.0, 5.0, 0.5, key="l_r", step=0.1)
            lc_reblow = st.slider("Reblow Risk Δ (%)", 0.0, 5.0, 1.0, key="l_rb", step=0.1)
            lc_clean = st.slider("Cleanliness Rejection Δ (%)", 0.000, 0.100, 0.010, format="%.3f", key="l_cl")
            lc_yield = st.slider("Yield Loss Δ (%)", 0.000, 0.100, 0.015, format="%.3f", key="l_y")
            lc_rh = st.slider("RH / Carbon Correction Δ (%)", 0.0, 20.0, 10.0, key="l_rh", step=0.5)
            
            lc_penalty = compute_operational_penalty(lc_thermal, lc_cycle, lc_rec_diff, lc_reblow, lc_clean, lc_yield, lc_rh, lc_eff, lc_price)
            st.info(f"**Total Operational Penalty: ₹{lc_penalty:,.0f} / MT**")

        with ec3:
            st.markdown("**Mn Briquette Constraints**")
            bq_mn = st.slider("Briquette Mn %", 90.0, 100.0, 99.0) / 100
            bq_rec = st.slider("Briquette Recovery %", 85.0, 100.0, 97.0) / 100
            bq_fe = 0.0
            bq_carbon = st.number_input("Briquette Carbon %", value=0.03, step=0.01)
            bq_eff = bq_mn * bq_rec
            bq_penalty = 0  # Baseline
            st.info(f"**Total Operational Penalty: ₹0 / MT (Baseline)**")

    def calc_tco(price, fe, penalty, eff_mn):
        fe_credit = fe * scrap_price
        adj_price = price - fe_credit
        base_viu = adj_price / eff_mn
        # FIX: Directly apply operational penalty without dividing by eff_mn
        penalty_per_eff = penalty 
        tco = base_viu + penalty_per_eff
        return fe_credit, adj_price, base_viu, penalty_per_eff, tco

    mc_cred, mc_adj, mc_base, mc_pen_eff, mc_tco = calc_tco(mc_price, mc_fe, mc_penalty, mc_eff)
    lc_cred, lc_adj, lc_base, lc_pen_eff, lc_tco = calc_tco(lc_price, lc_fe, lc_penalty, lc_eff)
    bq_cred, bq_adj, bq_base, bq_pen_eff, bq_tco = calc_tco(bq_price, bq_fe, bq_penalty, bq_eff)

    if active_alloys:
        tco_dict = {"MC FeMn": mc_tco, "LC FeMn": lc_tco, "Mn Briquette": bq_tco}
        active_tco_dict = {k: v for k, v in tco_dict.items() if k in active_alloys}
        best_alloy = min(active_tco_dict, key=active_tco_dict.get)
        st.success(f"### 🏆 Procurement Recommendation: **{best_alloy}** \n Lowest True Cost at **₹{min(active_tco_dict.values()):,.0f}** per MT of Effective Manganese.")

        # ── Comparison Heatmap ──
        df_tco = pd.DataFrame({
            "Metric": ["Base Price (₹/MT)", "Fe Credit (-)", "Adjusted Price (=)", "Effective Mn Content (%)", "Base VIU Cost (₹/MT Eff. Mn)", "Total Operational Penalty (+)", "Final TCO (₹/MT Eff. Mn)"],
            "MC FeMn": [mc_price, mc_cred, mc_adj, f"{mc_eff*100:.1f}%", round(mc_base), round(mc_pen_eff), round(mc_tco)],
            "LC FeMn": [lc_price, lc_cred, lc_adj, f"{lc_eff*100:.1f}%", round(lc_base), round(lc_pen_eff), round(lc_tco)],
            "Mn Briquette": [bq_price, bq_cred, bq_adj, f"{bq_eff*100:.1f}%", round(bq_base), round(bq_pen_eff), round(bq_tco)]
        })
        
        df_tco = df_tco[["Metric"] + active_alloys].set_index("Metric")

        st.markdown("#### 2. Cost Breakdown Heatmap")
        def highlight_min_tco(s):
            is_min = s == s.min()
            return ['background-color: #D4EDDA; color: #155724; font-weight: bold' if v else '' for v in is_min]

        df_numeric = df_tco.copy()
        for col in df_numeric.columns: df_numeric[col] = pd.to_numeric(df_numeric[col].astype(str).str.replace('%',''), errors='coerce')
        st.dataframe(df_tco.style.apply(highlight_min_tco, subset=pd.IndexSlice[["Final TCO (₹/MT Eff. Mn)", "Base VIU Cost (₹/MT Eff. Mn)"], :], axis=1).format(precision=0), use_container_width=True)

        # ── Visual Breakdown ──
        fig_tco = go.Figure()
        bases_map = {"MC FeMn": mc_base, "LC FeMn": lc_base, "Mn Briquette": bq_base}
        pens_map = {"MC FeMn": mc_pen_eff, "LC FeMn": lc_pen_eff, "Mn Briquette": bq_pen_eff}
        
        active_bases = [bases_map[a] for a in active_alloys]
        active_pens = [pens_map[a] for a in active_alloys]

        fig_tco.add_trace(go.Bar(name='Base VIU Cost (Adj Price / Eff Mn)', x=active_alloys, y=active_bases, marker_color='#2196F3', text=[f"₹{b:,.0f}" for b in active_bases], textposition='inside'))
        fig_tco.add_trace(go.Bar(name='Hidden Operational Penalty', x=active_alloys, y=active_pens, marker_color='#F44336', text=[f"₹{p:,.0f}" if p>0 else "" for p in active_pens], textposition='inside'))

        fig_tco.update_layout(barmode='stack', title="Total Cost of Ownership Breakdown (₹ per MT of Effective Mn)", template="plotly_white", height=450, yaxis_title="True Cost (₹/MT)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_tco, use_container_width=True)

        # ── Enterprise Savings Calculator ──
        st.markdown("#### 3. Enterprise Savings Calculator")
        sc1, sc2, sc3 = st.columns(3)
        baseline_alloy = sc1.selectbox("Current Baseline Practice", active_alloys, index=0)
        annual_mt = sc2.number_input("Yearly Consumption of Effective Mn (MT)", value=10000, step=1000)
        
        diff_per_mt = active_tco_dict[baseline_alloy] - active_tco_dict[best_alloy]
        savings_cr = (diff_per_mt * annual_mt) / 10000000
        
        sc3.metric(f"Savings vs {baseline_alloy} (₹/MT)", f"₹{diff_per_mt:,.0f}")
        
        if diff_per_mt > 0:
            st.info(f"### 💰 Projected Annual Savings: **₹{savings_cr:,.2f} Cr**")
        else:
            st.info(f"### 💰 Projected Annual Savings: **₹0.00 Cr** (Already using optimal alloy)")

        # ── 3-Year Future TCO Projections ──
        st.markdown("#### 4. Future TCO Projections (3-Year Forecast)")
        st.caption("Applies your selected VIU parameters (above) onto the ML engine's future market forecasts to project long-term true cost efficiency.")
        
        fut_mc = get_future_prices("./outputs_MC")
        fut_lc = get_future_prices("./outputs_LC")
        fut_bq = get_future_prices("./outputs_Briquette")
        
        if fut_mc is not None and fut_lc is not None and fut_bq is not None:
            fut_df = pd.DataFrame({
                "MC FeMn": fut_mc,
                "LC FeMn": fut_lc,
                "Mn Briquette": fut_bq
            }).dropna()
            
            # Recalculate TCO directly on the future time series
            mc_tco_series = ((fut_df["MC FeMn"] - (mc_fe * scrap_price)) / mc_eff) + mc_penalty
            lc_tco_series = ((fut_df["LC FeMn"] - (lc_fe * scrap_price)) / lc_eff) + lc_penalty
            bq_tco_series = ((fut_df["Mn Briquette"] - (bq_fe * scrap_price)) / bq_eff) + bq_penalty
            
            fig_fut_tco = go.Figure()
            
            if "MC FeMn" in active_alloys:
                fig_fut_tco.add_trace(go.Scatter(x=fut_df.index, y=mc_tco_series, name="MC FeMn TCO", line=dict(color="#FF9800", width=2.5)))
            if "LC FeMn" in active_alloys:
                fig_fut_tco.add_trace(go.Scatter(x=fut_df.index, y=lc_tco_series, name="LC FeMn TCO", line=dict(color="#2196F3", width=2.5)))
            if "Mn Briquette" in active_alloys:
                fig_fut_tco.add_trace(go.Scatter(x=fut_df.index, y=bq_tco_series, name="Mn Briquette TCO", line=dict(color="#4CAF50", width=2.5)))
                
            fig_fut_tco.update_layout(**_layout("Projected Total Cost of Ownership (Next 3 Years)", "True Cost (₹/MT Eff. Mn)", 450))
            st.plotly_chart(fig_fut_tco, use_container_width=True)
        else:
            st.info("📂 Future forecast data for all 3 alloys is required to generate this graph. Please run the pipeline module for all materials first.")

    else:
        st.warning("Please select at least one alloy to display the TCO comparison.")

# ══ TAB 5: LINEAR PROGRAMMING SOLVER ════════════════════════════════════════
with tab5:
    st.markdown("### 🧠 Optimal Alloy Substitution Solver")
    st.caption("Uses Linear Programming algorithms to calculate the mathematically cheapest blend of alloys that perfectly satisfy strict metallurgical limits (Carbon, Cleanliness, Recovery & Reblows).")

    blend_alloys = st.multiselect("Select Exactly TWO Alloys to Blend", ["MC FeMn", "LC FeMn", "Mn Briquette"], default=["LC FeMn", "Mn Briquette"])
    
    if len(blend_alloys) != 2:
        st.warning("⚠️ Please select exactly **TWO** alloys to calculate the optimal substitution mix.")
    else:
        # Dynamic Grade Limits driven by MC vs LC tables
        has_mc = "MC FeMn" in blend_alloys
        if has_mc:
            grades = {
                "Commodity Structural (IS2062/E250)": {"eff_mn": 0.65, "c_lim": 1.00, "rec_lim": 6.0, "reblow_lim": 5.0, "cln_lim": 0.12, "rh_lim": 2.5},
                "TMT/Rebar (Fe500D)":                 {"eff_mn": 0.70, "c_lim": 0.80, "rec_lim": 5.0, "reblow_lim": 4.0, "cln_lim": 0.10, "rh_lim": 2.0},
                "HSLA (API X60/X70)":                 {"eff_mn": 0.78, "c_lim": 0.50, "rec_lim": 3.0, "reblow_lim": 3.0, "cln_lim": 0.07, "rh_lim": 1.5},
                "Automotive Structural (DP600/780)":  {"eff_mn": 0.88, "c_lim": 0.35, "rec_lim": 2.0, "reblow_lim": 2.0, "cln_lim": 0.05, "rh_lim": 1.0},
                "Electrical Steel (CRGO/CRNO)":       {"eff_mn": 0.92, "c_lim": 0.20, "rec_lim": 1.5, "reblow_lim": 1.5, "cln_lim": 0.025, "rh_lim": 0.5},
                "IF Steel (Deep Draw IF)":            {"eff_mn": 0.95, "c_lim": 0.15, "rec_lim": 1.0, "reblow_lim": 1.0, "cln_lim": 0.03, "rh_lim": 0.5}
            }
        else:
            grades = {
                "Commodity Structural (IS2062/E250)": {"eff_mn": 0.74, "c_lim": 1.00, "rec_lim": 5.0, "reblow_lim": 4.0, "cln_lim": 0.10, "rh_lim": 2.0},
                "TMT/Rebar (Fe500D)":                 {"eff_mn": 0.76, "c_lim": 0.80, "rec_lim": 4.0, "reblow_lim": 3.5, "cln_lim": 0.09, "rh_lim": 1.8},
                "HSLA (API X60/X70)":                 {"eff_mn": 0.80, "c_lim": 0.50, "rec_lim": 3.0, "reblow_lim": 3.0, "cln_lim": 0.07, "rh_lim": 1.5},
                "Automotive Structural (DP600/780)":  {"eff_mn": 0.90, "c_lim": 0.35, "rec_lim": 2.0, "reblow_lim": 2.0, "cln_lim": 0.05, "rh_lim": 1.0},
                "API High-End (X80/X100)":            {"eff_mn": 0.85, "c_lim": 0.30, "rec_lim": 1.5, "reblow_lim": 1.5, "cln_lim": 0.04, "rh_lim": 0.5},
                "Electrical Steel (CRGO/CRNO)":       {"eff_mn": 0.93, "c_lim": 0.20, "rec_lim": 1.5, "reblow_lim": 1.5, "cln_lim": 0.025, "rh_lim": 0.5},
                "IF Steel (Deep Draw IF)":            {"eff_mn": 0.95, "c_lim": 0.15, "rec_lim": 1.0, "reblow_lim": 1.0, "cln_lim": 0.03, "rh_lim": 0.5}
            }

        gc1, gc2 = st.columns([1.5, 2.5])
        sel_grade = gc1.selectbox("Select Target Steel Grade", list(grades.keys()))
        
        st.markdown("#### Metallurgical Constraints")
        sc0, sc1, sc2, sc3, sc4, sc5 = st.columns(6)
        min_eff_mn = sc0.slider("Target Eff Mn (%)", 0.50, 1.00, grades[sel_grade]["eff_mn"], step=0.01)
        max_c = sc1.slider("Max Carbon (%)", 0.01, 1.50, grades[sel_grade]["c_lim"], step=0.01)
        max_r = sc2.slider("Max Reblows (%)", 0.5, 6.0, grades[sel_grade]["reblow_lim"], step=0.5)
        max_rec = sc3.slider("Rec Stability (±%)", 0.5, 6.0, grades[sel_grade]["rec_lim"], step=0.5)
        max_cln = sc4.slider("Cleanliness Limit", 0.01, 0.20, grades[sel_grade]["cln_lim"], step=0.01)
        max_rh  = sc5.slider("RH Max (min)", 0.0, 3.0, grades[sel_grade]["rh_lim"], step=0.1)

        # Calculate actual % Carbon added to steel per 1 MT of Effective Mn
        carb_mc = (mc_carbon / mc_eff) 
        carb_lc = (lc_carbon / lc_eff) 
        carb_bq = (bq_carbon / bq_eff) 

        # Constraint Mapping
        carb_map = {"MC FeMn": carb_mc, "LC FeMn": carb_lc, "Mn Briquette": carb_bq}
        tco_map = {"MC FeMn": mc_tco, "LC FeMn": lc_tco, "Mn Briquette": bq_tco}
        eff_mn_map = {"MC FeMn": mc_eff, "LC FeMn": lc_eff, "Mn Briquette": bq_eff}
        
        rec_map = {"MC FeMn": 5.0, "LC FeMn": 3.0, "Mn Briquette": 1.5}
        cln_map = {"MC FeMn": 0.12, "LC FeMn": 0.10, "Mn Briquette": 0.02}
        rh_map = {"MC FeMn": 2.5, "LC FeMn": 2.0, "Mn Briquette": 1.0}
        reblow_map = {"MC FeMn": 5.0, "LC FeMn": 3.0, "Mn Briquette": 1.0}

        a1, a2 = blend_alloys[0], blend_alloys[1]

        # ── LINEAR PROGRAMMING ENGINE ──
        c_cost = [tco_map[a1], tco_map[a2]]
        A_eq = [[1, 1]]
        b_eq = [1]
        
        A_ub = [
            [-eff_mn_map[a1], -eff_mn_map[a2]], # Effective Mn Constraint (negative to simulate >=)
            [carb_map[a1], carb_map[a2]],
            [reblow_map[a1], reblow_map[a2]],
            [rec_map[a1], rec_map[a2]],
            [cln_map[a1], cln_map[a2]],
            [rh_map[a1], rh_map[a2]]
        ]
        b_ub = [-min_eff_mn, max_c, max_r, max_rec, max_cln, max_rh]

        res = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1), (0, 1)])

        st.markdown("#### Optimization Result")
        if res.success:
            mix = res.x
            blended_tco = res.fun
            
            valid_singles = []
            if (eff_mn_map[a1] >= min_eff_mn) and (carb_map[a1] <= max_c) and (reblow_map[a1] <= max_r) and (rec_map[a1] <= max_rec) and (cln_map[a1] <= max_cln) and (rh_map[a1] <= max_rh): 
                valid_singles.append(tco_map[a1])
            if (eff_mn_map[a2] >= min_eff_mn) and (carb_map[a2] <= max_c) and (reblow_map[a2] <= max_r) and (rec_map[a2] <= max_rec) and (cln_map[a2] <= max_cln) and (rh_map[a2] <= max_rh): 
                valid_singles.append(tco_map[a2])
            
            baseline_cost = min(valid_singles) if valid_singles else max(tco_map.values())
            savings = baseline_cost - blended_tco

            rc1, rc2 = st.columns(2)
            rc1.success(f"##### Mathematically Optimal Blended TCO: \n ### **₹{blended_tco:,.0f}** per MT Eff. Mn")
            
            if savings > 10 and valid_singles:
                rc2.info(f"##### Projected Savings vs 100% Single Alloy: \n ### **₹{savings:,.0f}** per MT Eff. Mn")
            else:
                rc2.info(f"##### Projected Savings via Optimal Blending: \n ### **₹0** (100% Single Alloy is best)")

            pie_colors = {"MC FeMn": "#FF9