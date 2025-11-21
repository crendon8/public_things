"""
Utilities for generating fake daily fills data and plotting them with Plotly.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_NAMES: Sequence[str] = ("ABC", "DEF", "XYZ")


def create_fake_daily_frame(
    date: pd.Timestamp | str | None = None,
    names: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    sim_low: int = 80,
    sim_high: int = 140,
    prod_noise_mean: float = 0.0,
    prod_noise_sd: float = 10.0,
) -> pd.DataFrame:
    """
    Build a single day's dataframe without a date column.

    If rng is not provided, a deterministic seed is derived from `seed` or the date.
    """
    names = DEFAULT_NAMES if names is None else list(names)
    if rng is None:
        if seed is None and date is not None:
            seed = int(pd.to_datetime(date).value % (2**32))
        rng = np.random.default_rng(seed)
    sim_fills = rng.integers(sim_low, sim_high, size=len(names))
    prod_fills = sim_fills + rng.normal(prod_noise_mean, prod_noise_sd, size=len(names))
    pct_diff = (prod_fills - sim_fills) / sim_fills * 100
    df = pd.DataFrame(
        {
            "sim_fills": sim_fills.astype(int),
            "prod_fills": prod_fills.round(2),
            "pct_diff": pct_diff.round(2),
        },
        index=names,
    )
    df.index.name = "name"
    return df


def generate_fake_daily_frames(
    start: str = "2024-01-01",
    periods: int = 14,
    freq: str = "D",
    names: Sequence[str] | None = None,
    seed: int | None = 42,
    sim_low: int = 80,
    sim_high: int = 140,
    prod_noise_mean: float = 0.0,
    prod_noise_sd: float = 10.0,
) -> tuple[pd.DatetimeIndex, list[pd.DataFrame]]:
    """
    Create reproducible per-day dataframes (no date column) alongside the dates used.

    Returns dates, frames where each frame has index=name and sim/prod/pct_diff columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq=freq)
    frames = [
        create_fake_daily_frame(
            date=date,
            names=names,
            rng=rng,
            sim_low=sim_low,
            sim_high=sim_high,
            prod_noise_mean=prod_noise_mean,
            prod_noise_sd=prod_noise_sd,
        )
        for date in dates
    ]
    return dates, frames


def concat_daily_frames(frames: Iterable[pd.DataFrame], dates: Iterable[pd.Timestamp | str] | None = None) -> pd.DataFrame:
    """
    Combine daily dataframes into one table for plotting, attaching dates if provided.
    """
    frames = list(frames)
    prepared = []
    if dates is not None:
        dates = list(dates)
        if len(frames) != len(dates):
            raise ValueError("frames and dates must have the same length")
    for idx, frame in enumerate(frames):
        frame_with_date = frame.copy()
        if dates is not None:
            frame_with_date["date"] = pd.to_datetime(dates[idx])
        prepared.append(frame_with_date.reset_index())
    return pd.concat(prepared, ignore_index=True)


def plot_fills(df: pd.DataFrame, name: str) -> go.Figure:
    """
    Plot sim vs production fills over time with pct_diff on a secondary axis.
    """
    subset = df[df["name"] == name].sort_values("date")
    if subset.empty:
        raise ValueError(f"No rows found for name {name}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["sim_fills"],
            name="sim_fills",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["prod_fills"],
            name="prod_fills",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=subset["date"],
            y=subset["pct_diff"],
            name="pct_diff (%)",
            mode="lines+markers",
            line=dict(dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Sim vs Prod fills for {name}",
        hovermode="x unified",
        legend_title="Series",
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Fills", secondary_y=False)
    fig.update_yaxes(title_text="pct_diff (%)", secondary_y=True)
    return fig


__all__ = ["create_fake_daily_frame", "generate_fake_daily_frames", "concat_daily_frames", "plot_fills"]
