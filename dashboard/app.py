"""Streamlit dashboard for the four-hour energy cost recommendation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ENERGY_COLUMN,
    MODEL_PATH,
    PRICE_COLUMN,
    TIMESTAMP_COLUMN,
)
from src.data_loader import load_data  # noqa: E402
from src.forecasting import load_model, predict_energy_usage, train_model  # noqa: E402
from src.optimizer import (  # noqa: E402
    PREDICTED_ENERGY_COLUMN,
    calculate_window_cost,
    optimize_day_schedule,
)


PROCESS_HOURS = 4
BASELINE_START = 8
BASELINE_END = 12


st.set_page_config(
    page_title="Energy Window Optimizer",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp { background: #f6f8f7; }
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e2e8e5;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 3px 12px rgba(17, 48, 38, .04);
        }
        .recommendation {
            background: linear-gradient(120deg, #0b3d2e, #116149);
            color: white;
            border-radius: 18px;
            padding: 24px 28px;
            margin: 8px 0 18px 0;
            box-shadow: 0 8px 24px rgba(11, 61, 46, .16);
        }
        .recommendation .eyebrow {
            color: #9ce4c7; font-size: .78rem; font-weight: 700;
            letter-spacing: .08em; text-transform: uppercase;
        }
        .recommendation .window {
            font-size: 2rem; font-weight: 750; margin: 4px 0;
        }
        .recommendation .detail { color: #d7eee5; }
        .saving { color: #76e3b7; font-weight: 750; }
        .small-note { color: #65756f; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner=False)
def get_forecast_model() -> tuple[object, str]:
    """Load the persisted model, retraining in memory if it is incompatible."""
    try:
        return load_model(), "Saved random forest"
    except Exception:
        return train_model(get_data()), "Random forest retrained in memory"


@st.cache_data(show_spinner="Building the energy forecast…")
def add_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    model, model_label = get_forecast_model()
    result = predict_energy_usage(model, df)
    result[PREDICTED_ENERGY_COLUMN] = result[PREDICTED_ENERGY_COLUMN].clip(lower=0)
    return result, model_label


def available_windows(day_df: pd.DataFrame, earliest: int, latest: int) -> pd.DataFrame:
    """Return every valid four-hour window, ranked by forecast cost."""
    rows = []
    ordered = day_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    for index in range(len(ordered) - PROCESS_HOURS + 1):
        window = ordered.iloc[index : index + PROCESS_HOURS]
        start = window[TIMESTAMP_COLUMN].iloc[0]
        end = window[TIMESTAMP_COLUMN].iloc[-1] + pd.Timedelta(hours=1)
        if start.hour < earliest or end.hour > latest:
            continue
        rows.append(
            {
                "Start": start,
                "End": end,
                "Forecast cost (SEK)": calculate_window_cost(window),
                "Average price (SEK/kWh)": window[PRICE_COLUMN].mean(),
                "Forecast energy (kWh)": window[PREDICTED_ENERGY_COLUMN].sum(),
            }
        )
    return pd.DataFrame(rows).sort_values("Forecast cost (SEK)").reset_index(drop=True)


def draw_forecast_chart(
    day_df: pd.DataFrame,
    recommended_start: pd.Timestamp,
    recommended_end: pd.Timestamp,
) -> plt.Figure:
    fig, price_axis = plt.subplots(figsize=(12, 4.3))
    energy_axis = price_axis.twinx()
    timestamps = day_df[TIMESTAMP_COLUMN]

    price_axis.plot(
        timestamps,
        day_df[PRICE_COLUMN],
        color="#12745a",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label="Electricity price",
    )
    energy_axis.plot(
        timestamps,
        day_df[PREDICTED_ENERGY_COLUMN],
        color="#e39a32",
        linewidth=1.9,
        linestyle="--",
        label="Forecast energy",
    )
    price_axis.axvspan(
        recommended_start,
        recommended_end,
        color="#68d7aa",
        alpha=.22,
        label="Recommended window",
    )
    price_axis.set_ylabel("Price (SEK/kWh)", color="#12745a")
    energy_axis.set_ylabel("Forecast energy (kWh)", color="#b46c0a")
    # Pin the axis to the selected day. Without explicit limits Matplotlib adds
    # padding before midnight and can display 23:00 from the previous day.
    price_axis.set_xlim(timestamps.iloc[0], timestamps.iloc[-1])
    price_axis.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
    price_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    price_axis.grid(axis="y", alpha=.18)
    price_axis.spines[["top", "right"]].set_visible(False)
    energy_axis.spines["top"].set_visible(False)
    handles_1, labels_1 = price_axis.get_legend_handles_labels()
    handles_2, labels_2 = energy_axis.get_legend_handles_labels()
    price_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", frameon=False, ncol=3)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    return fig


raw_df = get_data()
forecast_df, model_name = add_forecast(raw_df)
available_dates = sorted(forecast_df[TIMESTAMP_COLUMN].dt.date.unique())

with st.sidebar:
    st.header("Planning controls")
    selected_date = st.selectbox(
        "Forecast date",
        available_dates,
        index=len(available_dates) - 1,
        format_func=lambda value: value.strftime("%A, %d %B %Y"),
    )
    earliest_start = st.slider("Earliest start", 0, 20, 6, format="%d:00")
    latest_end = st.slider("Latest finish", 4, 24, 22, format="%d:00")
    st.caption("The production run is fixed at 4 consecutive hours.")
    st.divider()
    st.caption(f"Forecast model: {model_name}")
    st.caption(f"Model file: `{MODEL_PATH.name}`")

st.title("⚡ Energy Window Optimizer")
st.caption("Schedule a four-hour production run when forecast energy costs are lowest.")

if latest_end - earliest_start < PROCESS_HOURS:
    st.error("The allowed planning window must be at least four hours wide.")
    st.stop()

day_df = forecast_df[
    forecast_df[TIMESTAMP_COLUMN].dt.date == selected_date
].copy()

try:
    result = optimize_day_schedule(
        day_df,
        process_duration_hours=PROCESS_HOURS,
        earliest_start=earliest_start,
        latest_end=latest_end,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

recommended_start = pd.Timestamp(result["recommended_start"])
recommended_end = pd.Timestamp(result["recommended_end"])
baseline_start = pd.Timestamp(selected_date) + pd.Timedelta(hours=BASELINE_START)
baseline_end = pd.Timestamp(selected_date) + pd.Timedelta(hours=BASELINE_END)

st.markdown(
    f"""
    <div class="recommendation">
      <div class="eyebrow">Recommended four-hour start window</div>
      <div class="window">{recommended_start:%H:%M} → {recommended_end:%H:%M}</div>
      <div class="detail">
        Forecast cost <b>{result['estimated_cost']:,.2f} SEK</b> ·
        save <span class="saving">{result['saving']:,.2f} SEK ({result['saving_percent']:.1f}%)</span>
        versus the 08:00–12:00 schedule
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Optimized cost", f"{result['estimated_cost']:,.2f} SEK")
metric_2.metric("Baseline cost", f"{result['baseline_cost']:,.2f} SEK")
metric_3.metric(
    "Savings",
    f"{result['saving']:,.2f} SEK",
    delta=f"{result['saving_percent']:.1f}% lower",
    delta_color="normal",
)

recommended_rows = day_df[
    (day_df[TIMESTAMP_COLUMN] >= recommended_start)
    & (day_df[TIMESTAMP_COLUMN] < recommended_end)
]
baseline_rows = day_df[
    (day_df[TIMESTAMP_COLUMN] >= baseline_start)
    & (day_df[TIMESTAMP_COLUMN] < baseline_end)
]
recommended_avg_price = recommended_rows[PRICE_COLUMN].mean()
baseline_avg_price = baseline_rows[PRICE_COLUMN].mean()
metric_4.metric(
    "Average window price",
    f"{recommended_avg_price:.3f} SEK/kWh",
    delta=f"{recommended_avg_price - baseline_avg_price:.3f} vs baseline",
    delta_color="inverse",
)

st.subheader("Price and energy forecast")
st.pyplot(
    draw_forecast_chart(day_df, recommended_start, recommended_end),
    width="stretch",
)
st.markdown(
    '<div class="small-note">The shaded region is the recommended run window. '
    "Cost combines hourly electricity price with the model's forecast energy use.</div>",
    unsafe_allow_html=True,
)

rankings = available_windows(day_df, earliest_start, latest_end)
left, right = st.columns([1.15, .85])
with left:
    st.subheader("Best alternatives")
    display_rankings = rankings.head(5).copy()
    display_rankings.insert(0, "Rank", range(1, len(display_rankings) + 1))
    display_rankings["Window"] = display_rankings.apply(
        lambda row: f"{row['Start']:%H:%M}–{row['End']:%H:%M}", axis=1
    )
    st.dataframe(
        display_rankings[
            ["Rank", "Window", "Forecast cost (SEK)", "Average price (SEK/kWh)"]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Forecast cost (SEK)": st.column_config.NumberColumn(format="%.2f"),
            "Average price (SEK/kWh)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

with right:
    st.subheader("Operational snapshot")
    forecast_energy = recommended_rows[PREDICTED_ENERGY_COLUMN].sum()
    price_peak = day_df.loc[day_df[PRICE_COLUMN].idxmax()]
    st.markdown(
        f"""
        - **Forecast consumption:** {forecast_energy:,.1f} kWh during the run
        - **Daily price peak:** {price_peak[TIMESTAMP_COLUMN]:%H:%M} at {price_peak[PRICE_COLUMN]:.3f} SEK/kWh
        - **Baseline compared:** {baseline_start:%H:%M}–{baseline_end:%H:%M}
        - **Valid 4-hour windows compared:** {len(rankings)}
        """
    )

with st.expander("View hourly forecast data"):
    detail_df = day_df[
        [TIMESTAMP_COLUMN, PRICE_COLUMN, ENERGY_COLUMN, PREDICTED_ENERGY_COLUMN]
    ].copy()
    detail_df["Recommended"] = detail_df[TIMESTAMP_COLUMN].between(
        recommended_start, recommended_end, inclusive="left"
    )
    st.dataframe(
        detail_df,
        hide_index=True,
        width="stretch",
        column_config={
            TIMESTAMP_COLUMN: st.column_config.DatetimeColumn("Hour", format="HH:mm"),
            PRICE_COLUMN: st.column_config.NumberColumn("Price (SEK/kWh)", format="%.3f"),
            ENERGY_COLUMN: st.column_config.NumberColumn("Historical usage (kWh)", format="%.1f"),
            PREDICTED_ENERGY_COLUMN: st.column_config.NumberColumn("Forecast usage (kWh)", format="%.1f"),
            "Recommended": st.column_config.CheckboxColumn("In recommended window"),
        },
    )

st.caption(
    "Decision-support estimate based on simulated prices and modelled energy use; "
    "confirm operational constraints before scheduling production."
)
