"""
UNIFIED MANGANESE FORECASTING DASHBOARD
=================================================
Handles Mn Briquette (97%), LC FeMn (80%), and MC FeMn (70%).
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

st.markdown(f"## 📈 {alloy_choice} Price Forecasting Engine")

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

tab1, tab2, tab3, tab4 = st.tabs(["📉 Price Forecast", "📊 Market Comparison", "🔀 Regime & Drivers", "🏭 Industrial UI"])

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
    
    driver_list = ["simn", "mn_ore", "chn_electricity", "met_coal", "dry_bulk_freight", "steel_etf", "usd_inr"]
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

# ══ TAB 4: DYNAMIC INDUSTRIAL CALCULATOR ═════════════════════════════════════
with tab4:
    h_viu, f_viu, fig3 = h.copy(), future.copy(), go.Figure()
    
    if alloy_code == "Briquette":
        st.markdown("### 🏭 Briquetting Penalty & True Cost")
        st.caption("The algorithm outputs the price of Mn Flakes. This calculates the true cost of making Briquettes.")
        cc1, cc2 = st.columns(2)
        base_mult = cc1.number_input("Base Flake Multiplier", value=1.00, step=0.01)
        proc_cost = cc2.number_input("Briquetting Processing Cost (Rs/Kg)", value=3.00, step=0.50)
        
        h_viu["viu"] = (h_viu["real_price"] * base_mult) + proc_cost
        f_viu["viu"] = (f_viu["real_price"] * base_mult) + proc_cost
        
        fig3.add_trace(go.Scatter(x=full_x, y=list(h_viu["real_price"]) + [None]*M, name="Base Flake Price", line=dict(color=C_HYBRID, width=3)))
        fig3.add_trace(go.Scatter(x=full_x, y=list(h_viu["viu"]) + [None]*M, name="True Briquette Cost", line=dict(color=C_ACTUAL, width=2.5, dash="dash"), yaxis="y2"))
        fig3.add_trace(go.Scatter(x=full_x, y=[None]*(N-1) + [h_viu["real_price"].iloc[-1]] + list(f_viu["real_price"]), name="Future Flake Price", line=dict(color=C_FUTURE, width=3)))
        fig3.add_trace(go.Scatter(x=full_x, y=[None]*(N-1) + [h_viu["viu"].iloc[-1]] + list(f_viu["viu"]), name="Future Briquette Cost", line=dict(color="#FF5722", width=2.5, dash="dash"), yaxis="y2"))
        fig3.update_layout(**_layout(f"Base Flake Price vs. True Briquette Cost (+{proc_cost} Rs/Kg)", "Flake Price (Rs/Kg)", 450))
        fig3.update_layout(yaxis2=dict(title="True Briquette Cost (Rs/Kg)", overlaying="y", side="right", showgrid=False, matches="y"))
        
    else:
        st.markdown("### 🏭 Value-in-Use (VIU) Calculator")
        st.caption("Calculates the true cost per kg of effective Manganese added to the steel bath, factoring in recovery rates.")
        cc1, cc2 = st.columns(2)
        mn_pct  = cc1.number_input("Manganese Content (%)", value=(80.0 if alloy_code == "LC" else 70.0), step=0.5)
        rec_pct = cc2.number_input("Recovery Rate (%)", value=(90.0 if alloy_code == "LC" else 85.0), step=0.5)
        
        eff_mn_frac = (mn_pct / 100.0) * (rec_pct / 100.0)
        h_viu["viu"] = h_viu["real_price"] / eff_mn_frac
        f_viu["viu"] = f_viu["real_price"] / eff_mn_frac
        
        fig3.add_trace(go.Scatter(x=full_x, y=list(h_viu["real_price"]) + [None]*M, name="Alloy Price (Raw)", line=dict(color=C_HYBRID, width=3)))
        fig3.add_trace(go.Scatter(x=full_x, y=list(h_viu["viu"]) + [None]*M, name="True VIU Cost", line=dict(color=C_ACTUAL, width=2.5, dash="dash"), yaxis="y2"))
        fig3.add_trace(go.Scatter(x=full_x, y=[None]*(N-1) + [h_viu["real_price"].iloc[-1]] + list(f_viu["real_price"]), name="Future Alloy Price", line=dict(color=C_FUTURE, width=3)))
        fig3.add_trace(go.Scatter(x=full_x, y=[None]*(N-1) + [h_viu["viu"].iloc[-1]] + list(f_viu["viu"]), name="Future VIU Cost", line=dict(color="#FF5722", width=2.5, dash="dash"), yaxis="y2"))
        fig3.update_layout(**_layout(f"Raw Price vs. True Metallurgical Cost (Mn={mn_pct}%, Rec={rec_pct}%)", "Raw Price (Rs/Kg)", 450))
        fig3.update_layout(yaxis2=dict(title="True Cost (Rs/Kg Effective Mn)", overlaying="y", side="right", showgrid=False, matches="y"))
        
    st.plotly_chart(fig3, use_container_width=True)