# app.py
import os, sys
import numpy as np
import pandas as pd
import torch
from argparse import Namespace

import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import warnings
warnings.filterwarnings('ignore')

from ml.utils.data_utils import prepare_dataset
from ml.models.seq2seq_lstm import Seq2SeqLSTM
from ml.models.transformer import TimeSeriesTransformer

# =========================
# Config
# =========================
st.set_page_config(page_title="Slice-aware FL Prediction", layout="wide")

# Updated paths
DATAPATH       = "../dataset/combined_with_cluster_feature_with_extraData_26Aug.csv"
CKPT_S2S_CLU   = "seq2seq_cluster_with_extra_data_26Aug.pt"
CKPT_TRANS_CLU = "transformer_multistep_cluster_with_extra_data_26Aug.pt"

TARGETS   = ["rnti_count", "rb_down", "rb_up", "down", "up"]
H         = 6               # horizon
L         = 10              # history window (lags)
DATA_FREQ = "2min"          # dataset cadence

# Map – Barcelona + placeholder RIY* points near Riyadh
STATIONS = {
    # Barcelona
    "ElBorn":   (41.3853, 2.1827),
    "LesCorts": (41.3850, 2.1247),
    "PobleSec": (41.3724, 2.1620),
    # Riyadh placeholders (lat, lon ~ city area)
    "RIY0329":  (24.7136, 46.6753),  # central
    "RIY0443":  (24.7255, 46.7002),  # north-east
    "RIY2110":  (24.6768, 46.6840),  # south-west
    "RIY2227":  (24.7379, 46.6370),  # north-west
}

# =========================
# Metrics (per-variable, as provided)
# =========================
# Keep short names that match the sidebar selection below
METRICS = pd.DataFrame([
    # Transformer
    ["Transformer", "rnti_count (Number of Users)",          0.0130, 0.1141, 0.0914, 0.2821, 0.1509,  38.22,  31.63],
    ["Transformer", "rb_down (Downlink Resource Blocks)",    0.0066, 0.0813, 0.0420, 0.5480, 0.0816,  48.44,  39.42],
    ["Transformer", "rb_up (Uplink Resource Blocks)",        0.0164, 0.1279, 0.0617, 0.4615, 0.1279, 2273.49, 113.19],
    ["Transformer", "down (Downlink Traffic)",               0.0042, 0.0650, 0.0517, 0.4305, 0.0797,  51.20,  36.48],
    ["Transformer", "up (Uplink Traffic)",                   0.0185, 0.1360, 0.0652, 0.3603, 0.1360, 1167.29,  95.99],
    # Seq2Seq
    ["Seq2Seq LSTM", "rnti_count (Number of Users)",         0.0153, 0.1236, 0.0945, 0.1567, 0.1635,  32.94,  32.96],
    ["Seq2Seq LSTM", "rb_down (Downlink Resource Blocks)",   0.0060, 0.0773, 0.0408, 0.5912, 0.0776,  47.21,  39.80],
    ["Seq2Seq LSTM", "rb_up (Uplink Resource Blocks)",       0.0176, 0.1326, 0.0623, 0.4205, 0.1326, 2689.38, 109.64],
    ["Seq2Seq LSTM", "down (Downlink Traffic)",              0.0034, 0.0586, 0.0430, 0.5382, 0.0717,  39.01,  30.69],
    ["Seq2Seq LSTM", "up (Uplink Traffic)",                  0.0202, 0.1422, 0.0654, 0.3002, 0.1422,  764.47,  98.90],
], columns=["Experiment","Prediction Variable","MSE","RMSE","MAE","R²","NRMSE","MAPE%","sMAPE%"])

# =========================
# Utilities
# =========================
def normalize_name(s: str) -> str:
    s0 = str(s).strip()
    s  = s0.lower()
    if "el born" in s or s.startswith("elborn"):      return "ElBorn"
    if "les corts" in s or s.startswith("lescorts"):  return "LesCorts"
    if "poble sec" in s or s.startswith("poblesec"):  return "PobleSec"
    # RIY* pass-through (ensure exact keys)
    if s0.upper().startswith("RIY"):
        return s0.upper()
    return s0.title()

def inverse_single_col(y_scaled_1d: np.ndarray, scaler, j: int) -> np.ndarray:
    y_scaled_1d = np.asarray(y_scaled_1d)
    if hasattr(scaler, "min_") and hasattr(scaler, "scale_"):      # MinMax
        return (y_scaled_1d - scaler.min_[j]) / scaler.scale_[j]
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):     # Standard
        return y_scaled_1d * scaler.scale_[j] + scaler.mean_[j]
    raise ValueError("Unknown scaler type for inverse transform.")

# =========================
# Cached loaders
# =========================
@st.cache_data(show_spinner=False)
def load_raw_and_scalers():
    raw = pd.read_csv(DATAPATH, parse_dates=["time"])
    raw["District"] = raw["District"].map(normalize_name)
    raw = raw.sort_values("time").reset_index(drop=True)

    args = Namespace(
        data_path=DATAPATH, targets=TARGETS, num_lags=L, forecast_steps=H,
        test_size=0.2, ignore_cols=None, identifier="District",
        nan_constant=0, x_scaler="minmax", y_scaler="minmax",
        outlier_detection=True, batch_size=128, cuda=torch.cuda.is_available(),
        seed=42, use_time_features=False,
    )
    X_tr, y_tr, X_te, y_te, x_scaler, y_scaler, id_tr, id_te = prepare_dataset(args)
    D = X_te.shape[2]

    X_num = raw.drop(columns=TARGETS, errors="ignore").select_dtypes(include=[np.number]).copy()
    x_base_cols = list(X_num.columns)

    bounds = {}
    off = pd.to_timedelta(DATA_FREQ)
    for station in STATIONS.keys():
        d = raw[raw["District"] == station].copy()
        if d.empty:
            bounds[station] = (None, None); continue
        d = d.reset_index(drop=True)

        Xb = d.drop(columns=TARGETS, errors="ignore").select_dtypes(include=[np.number]).copy()
        for c in x_base_cols:
            if c not in Xb.columns: Xb[c] = 0.0
        Xb = Xb[x_base_cols]
        Xb_scaled = pd.DataFrame(x_scaler.transform(Xb.values), columns=x_base_cols, index=d.index)

        X_lagged = pd.DataFrame(index=d.index)
        for col in x_base_cols:
            for lag in range(1, L + 1):
                X_lagged[f"{col}_lag_{lag}"] = Xb_scaled[col].shift(lag)

        y_future = pd.DataFrame(index=d.index)
        for i in range(1, H + 1):
            for tcol in TARGETS:
                y_future[f"{tcol}_step_{i}"] = d[tcol].shift(-i)

        glued = pd.concat([d[["time"]], X_lagged, y_future], axis=1).dropna().reset_index(drop=True)
        if glued.empty:
            bounds[station] = (None, None); continue

        last_future_times = glued["time"] + H * off
        bounds[station] = (last_future_times.min(), last_future_times.max())

    districts_all = sorted(set(raw["District"].dropna().unique()).intersection(set(STATIONS.keys())))
    return raw, x_scaler, y_scaler, D, x_base_cols, bounds, districts_all

@st.cache_resource(show_spinner=False)
def load_models(D):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s2s = Seq2SeqLSTM(input_size=D, hidden_size=64, output_size=len(TARGETS),
                      forecast_steps=H, num_layers=1).to(device)
    s2s.load_state_dict(torch.load(CKPT_S2S_CLU, map_location=device), strict=True)
    s2s.eval()

    trans = TimeSeriesTransformer(
        input_size=D, output_size=len(TARGETS), forecast_steps=H,
        d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=256, dropout=0.1
    ).to(device)
    trans.load_state_dict(torch.load(CKPT_TRANS_CLU, map_location=device), strict=True)
    trans.eval()

    return s2s, trans, device

# =========================
# Build windows (from raw)
# =========================
def build_windows_for_station(raw_df, station, sel_start, sel_end, x_base_cols, x_scaler):
    """
    Returns:
      X_scaled_3d: [N, L, D]  (scaled & lagged)
      Y_orig:      [N, H, T]  (future true, original scale)
      HIST_orig:   [N, L, T]  (history true, original scale, no overlap with future)
      last_future_times: list of timestamps (for info)
    """
    off = pd.to_timedelta(DATA_FREQ)
    d = raw_df[raw_df["District"] == station].copy()
    if d.empty:
        return (np.zeros((0, L, 0)), np.zeros((0, H, len(TARGETS))), np.zeros((0, L, len(TARGETS))), [])

    d = d.sort_values("time").reset_index(drop=True)

    Xb = d.drop(columns=TARGETS, errors="ignore").select_dtypes(include=[np.number]).copy()
    for c in x_base_cols:
        if c not in Xb.columns: Xb[c] = 0.0
    Xb = Xb[x_base_cols]
    Xb_scaled = pd.DataFrame(x_scaler.transform(Xb.values), columns=x_base_cols, index=d.index)

    X_lagged = pd.DataFrame(index=d.index)
    for col in x_base_cols:
        for lag in range(1, L + 1):
            X_lagged[f"{col}_lag_{lag}"] = Xb_scaled[col].shift(lag)

    y_future = pd.DataFrame(index=d.index)
    for i in range(1, H + 1):
        for tcol in TARGETS:
            y_future[f"{tcol}_step_{i}"] = d[tcol].shift(-i)

    glued = pd.concat([d[["time", *TARGETS]], X_lagged, y_future], axis=1).dropna().reset_index(drop=True)
    if glued.empty:
        D_here = Xb_scaled.shape[1]
        return (np.zeros((0, L, D_here)), np.zeros((0, H, len(TARGETS))), np.zeros((0, L, len(TARGETS))), [])

    last_future_times = glued["time"] + H * off
    mask = (last_future_times >= pd.to_datetime(sel_start)) & (last_future_times <= pd.to_datetime(sel_end))
    glued = glued.loc[mask].reset_index(drop=True)
    last_future_times = last_future_times.loc[mask].reset_index(drop=True)
    if glued.empty:
        D_here = Xb_scaled.shape[1]
        return (np.zeros((0, L, D_here)), np.zeros((0, H, len(TARGETS))), np.zeros((0, L, len(TARGETS))), [])

    lag_cols = [c for c in X_lagged.columns]
    X_flat = glued[lag_cols].values
    D_here = len(x_base_cols)
    X_scaled_3d = X_flat.reshape(-1, L, D_here)

    y_cols = []
    for i in range(1, H + 1):
        for tcol in TARGETS:
            y_cols.append(f"{tcol}_step_{i}")
    Y_orig = glued[y_cols].values.reshape(-1, H, len(TARGETS))

    d2 = d.set_index("time")
    HIST_orig = np.zeros((X_scaled_3d.shape[0], L, len(TARGETS)), dtype=float)
    for n, t_last in enumerate(pd.to_datetime(last_future_times)):
        t_first_future = t_last - (H - 1) * off
        hist_end = t_first_future - off
        hist_idx = pd.date_range(end=hist_end, periods=L, freq=off)
        block = d2[TARGETS].reindex(hist_idx).ffill().bfill()
        HIST_orig[n, :, :] = block.values

    return X_scaled_3d, Y_orig, HIST_orig, list(last_future_times)

# =========================
# Plot helpers
# =========================
def panel_fig(name, hist, fut_true, fut_pred):
    """Single panel for one target with required colors/styles."""
    Lh = hist.shape[0]; Hh = fut_true.shape[0]
    x_hist = np.arange(-Lh, 0, 1)
    x_fut  = np.arange(1, Hh + 1, 1)

    fig = go.Figure()
    # History (light blue)
    fig.add_trace(go.Scatter(x=x_hist, y=hist, mode="lines+markers",
                             name=f"{name} (history)", line=dict(color="#83C9FF")))
    # True (dark blue)
    fig.add_trace(go.Scatter(x=x_fut, y=fut_true, mode="lines+markers",
                             name=f"{name} (true)", line=dict(color="#0068C9")))
    # Pred (orange dotted)
    fig.add_trace(go.Scatter(x=x_fut, y=fut_pred, mode="lines+markers",
                             name=f"{name} (pred)", line=dict(color="#FF8C00", dash="dot")))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10),
                      title=name, xaxis_title="Relative time", yaxis_title="Original scale")
    return fig

def render_panels(HIST_orig, Y_orig, Y_pred_orig):
    figs = []
    if HIST_orig.shape[0] == 0:
        return figs
    hist_last = HIST_orig[-1]
    true_last = Y_orig[-1]
    pred_last = Y_pred_orig[-1]
    TARGET_NAMES = ["rnti (Number of Users)", "rb_down (Downlink Resource Blocks)",
                    "rb_up (Uplink Resource Blocks)", "down (Downlink Traffic)", "up (Uplink Traffic)"]
    for j, name in enumerate(TARGET_NAMES):
        figs.append(panel_fig(name, hist_last[:, j], true_last[:, j], pred_last[:, j]))
    return figs

# =========================
# UI
# =========================
st.markdown("## Slice-aware FL Multi-step Prediction Dashboard")

# Load raw + scalers + bounds
raw, x_scaler, y_scaler, D_feat, x_base_cols, time_bounds, districts_all = load_raw_and_scalers()

# Sidebar controls
with st.sidebar:
    st.markdown("### Controls")
    bs_choice = st.selectbox("Base Station", options=districts_all, index=0)
    model_choice = st.selectbox("FL Model", options=["Seq2Seq LSTM", "Transformer"], index=0)

    # Station-specific time bounds
    tmin, tmax = time_bounds.get(bs_choice, (None, None))
    if tmin is None or tmax is None:
        dtmp = raw[raw["District"] == bs_choice].copy()
        if not dtmp.empty:
            tmin, tmax = dtmp["time"].min(), dtmp["time"].max()
        else:
            tmin, tmax = raw["time"].min(), raw["time"].max()

    sel_range = st.slider(
        "Time Selection (based on last future time t+H)",
        min_value=pd.to_datetime(tmin).to_pydatetime(),
        max_value=pd.to_datetime(tmax).to_pydatetime(),
        value=(pd.to_datetime(tmin).to_pydatetime(), pd.to_datetime(tmax).to_pydatetime()),
        step=pd.to_timedelta(DATA_FREQ),
        format="YYYY-MM-DD HH:mm",
    )

# Metrics table (top-center): filter by selected model; white text on dark background
df_selected = METRICS[METRICS["Experiment"] == model_choice].copy()
st.dataframe(
    df_selected.style.set_properties(**{"color": "white", "background-color": "#1f2937"}),
    use_container_width=True
)

# Load models once
s2s_model, trans_model, DEVICE = load_models(D_feat)

# Build windows for selected station/time
X_scaled_3d, Y_orig, HIST_orig, last_future_times = build_windows_for_station(
    raw_df=raw,
    station=bs_choice,
    sel_start=pd.to_datetime(sel_range[0]),
    sel_end=pd.to_datetime(sel_range[1]),
    x_base_cols=x_base_cols,
    x_scaler=x_scaler
)

# Predict (scaled) then inverse to original scale
if X_scaled_3d.shape[0] > 0:
    xb = torch.tensor(X_scaled_3d, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        if model_choice.startswith("Seq2Seq"):
            y_pred_scaled = s2s_model(xb, teacher_forcing_ratio=0.0).cpu().numpy()
        else:
            y_pred_scaled = trans_model(xb).cpu().numpy()
    Nw, Hh, Tn = y_pred_scaled.shape
    Y_pred_orig = np.zeros_like(y_pred_scaled, dtype=float)
    for j in range(Tn):
        Y_pred_orig[:, :, j] = inverse_single_col(y_pred_scaled[:, :, j], y_scaler, j)
else:
    Y_pred_orig = np.zeros((0, H, len(TARGETS)))

# Layout: charts (left), map (right)
left, right = st.columns([1.25, 1.0], gap="large")

with right:
    st.markdown(f"### Base Station: `{bs_choice}`")

    # Build points (selected bright, others dim)
    pts = []
    for name, (lat, lon) in STATIONS.items():
        sel = (name == bs_choice)
        pts.append(dict(
            name=name,
            lat=lat,
            lon=lon,
            color=[0, 200, 140] if sel else [160, 160, 160],
            radius=300 if sel else 140,
        ))
    df_map = pd.DataFrame(pts)

    # Center the camera on the selected site
    sel_lat, sel_lon = STATIONS[bs_choice]

    st.pydeck_chart(pdk.Deck(
        map_provider="carto",
        map_style="light",  # "light", "dark-matter", or "voyager"
        initial_view_state=pdk.ViewState(
            latitude=float(sel_lat),
            longitude=float(sel_lon),
            zoom=13 if bs_choice.startswith("RIY") else 12,
            pitch=0,
            bearing=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position='[lon, lat]',
                get_radius="radius",
                get_fill_color="color",
                pickable=True,
            )
        ],
        tooltip={"text": "{name}"}
    ))

with left:
    if X_scaled_3d.shape[0] == 0:
        st.info(f"No complete windows in range for **{bs_choice}**. Try expanding time selection or switching site.")
    else:
        figs = render_panels(HIST_orig, Y_orig, Y_pred_orig)
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)
