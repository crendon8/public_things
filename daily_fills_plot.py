"""
Utilities for generating fake daily fills data and plotting them with Plotly.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_NAMES: Sequence[str] = ("ABC", "DEF", "XYZ")

GroupSpec = Mapping[str, Sequence[tuple[str, str]] | str]


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
    """
    names = DEFAULT_NAMES if names is None else list(names)
    if rng is None:
        if seed is None and date is not None:
            seed = int(pd.to_datetime(date).value % (2**32))
        rng = np.random.default_rng(seed)
    sim_fills = rng.integers(sim_low, sim_high, size=len(names))
    prod_fills = sim_fills + rng.normal(prod_noise_mean, prod_noise_sd, size=len(names))
    pct_diff = (prod_fills - sim_fills) / sim_fills * 100

    # Buy/sell breakdown
    sim_buy_fills = np.round(sim_fills * rng.uniform(0.4, 0.6, size=len(names))).astype(int)
    sim_sell_fills = (sim_fills - sim_buy_fills).astype(int)
    prod_buy_fills = np.round(sim_buy_fills + rng.normal(0, 5, size=len(names))).astype(int)
    prod_sell_fills = np.round(sim_sell_fills + rng.normal(0, 5, size=len(names))).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        total_buy_pct_diff = np.where(
            sim_buy_fills == 0, np.nan, (prod_buy_fills - sim_buy_fills) / sim_buy_fills * 100
        )
        total_sell_pct_diff = np.where(
            sim_sell_fills == 0, np.nan, (prod_sell_fills - sim_sell_fills) / sim_sell_fills * 100
        )

    # Count metrics
    sim_buys = rng.integers(40, 90, size=len(names))
    sim_sells = rng.integers(40, 90, size=len(names))
    prod_buys = sim_buys + rng.normal(0, 8, size=len(names))
    prod_sells = sim_sells + rng.normal(0, 8, size=len(names))
    with np.errstate(divide="ignore", invalid="ignore"):
        ct_buy_pct = np.where(sim_buys == 0, np.nan, (prod_buys - sim_buys) / sim_buys * 100)
        ct_sell_ct = np.where(sim_sells == 0, np.nan, (prod_sells - sim_sells) / sim_sells * 100)

    # Price metrics
    sim_buy_px = rng.normal(100, 5, size=len(names))
    sim_sell_px = rng.normal(101, 5, size=len(names))
    prod_buy_px = sim_buy_px + rng.normal(0, 0.5, size=len(names))
    prod_sell_px = sim_sell_px + rng.normal(0, 0.5, size=len(names))
    with np.errstate(divide="ignore", invalid="ignore"):
        buy_bps_diff = np.where(sim_buy_px == 0, np.nan, (prod_buy_px - sim_buy_px) / sim_buy_px * 10_000)
        sell_bps_diff = np.where(sim_sell_px == 0, np.nan, (prod_sell_px - sim_sell_px) / sim_sell_px * 10_000)

    df = pd.DataFrame(
        {
            "sim_fills": sim_fills.astype(int),
            "prod_fills": prod_fills.round(2),
            "pct_diff": pct_diff.round(2),
            "sim_buy_fills": sim_buy_fills,
            "sim_sell_fills": sim_sell_fills,
            "prod_buy_fills": prod_buy_fills,
            "prod_sell_fills": prod_sell_fills,
            "total_buy_pct_diff": np.round(total_buy_pct_diff, 2),
            "total_sell_pct_diff": np.round(total_sell_pct_diff, 2),
            "sim_buys": sim_buys.astype(int),
            "sim_sells": sim_sells.astype(int),
            "prod_buys": prod_buys.round(2),
            "prod_sells": prod_sells.round(2),
            "ct_buy_pct": np.round(ct_buy_pct, 2),
            "ct_sell_ct": np.round(ct_sell_ct, 2),
            "sim_buy_px": sim_buy_px.round(4),
            "sim_sell_px": sim_sell_px.round(4),
            "prod_buy_px": prod_buy_px.round(4),
            "prod_sell_px": prod_sell_px.round(4),
            "buy_bps_diff": np.round(buy_bps_diff, 2),
            "sell_bps_diff": np.round(sell_bps_diff, 2),
        },
        index=names,
    )
    df.index.name = "symbol"
    return df


def create_fake_pnl_frame(
    date: pd.Timestamp | str | None = None,
    names: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    sim_mean: float = 0.0,
    sim_sd: float = 5_000.0,
    prod_noise_mean: float = 0.0,
    prod_noise_sd: float = 1_500.0,
) -> pd.DataFrame:
    """
    Build a single day's PnL dataframe with sim/prod pnl and pct_diff for each symbol.
    """
    names = DEFAULT_NAMES if names is None else list(names)
    if rng is None:
        if seed is None and date is not None:
            seed = int(pd.to_datetime(date).value % (2**32))
        rng = np.random.default_rng(seed)
    sim_pnl = rng.normal(sim_mean, sim_sd, size=len(names))
    prod_pnl = sim_pnl + rng.normal(prod_noise_mean, prod_noise_sd, size=len(names))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_diff = np.where(sim_pnl == 0, np.nan, (prod_pnl - sim_pnl) / np.abs(sim_pnl) * 100)
    df = pd.DataFrame(
        {
            "sim_pnl": sim_pnl.round(2),
            "prod_pnl": prod_pnl.round(2),
            "pct_diff": np.round(pct_diff, 2),
        },
        index=names,
    )
    df.index.name = "symbol"
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
    """Combine daily dataframes into one table for plotting, attaching dates if provided."""
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


def plot_groups(
    df: pd.DataFrame,
    group: GroupSpec,
    add_dropdown: bool = True,
    symbol_mask: Sequence[str] | None = None,
) -> go.Figure:
    """
    Plot series for available symbols using the provided group definition. A dropdown toggles symbols.

    `group` should provide "primary" and/or "secondary" lists of (column, label) tuples.
    Optionally include a "title" string (supports {symbol} formatting).
    """
    symbols = sorted(df["symbol"].unique())
    if symbol_mask is not None:
        # An empty mask means no filtering; a non-empty mask limits to those names present.
        if symbol_mask:
            symbols = [s for s in symbols if s in symbol_mask]
    if not symbols:
        raise ValueError("No symbols found in dataframe")
    initial_symbol = symbols[0]

    primaries = list(group.get("primary", []))
    secondaries = list(group.get("secondary", []))
    required_cols = [col for col, _ in primaries] + [col for col, _ in secondaries]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for plotting: {missing}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    traces_per_symbol = len(primaries) + len(secondaries)
    for sym in symbols:
        subset = df[df["symbol"] == sym].sort_values("date").copy()
        subset["date"] = pd.to_datetime(subset["date"])
        visible = sym == initial_symbol
        for col, label in primaries:
            fig.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset[col],
                    name=label,
                    mode="lines+markers",
                    visible=visible,
                )
            )
        for col, label in secondaries:
            fig.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset[col],
                    name=label,
                    mode="lines+markers",
                    visible=visible,
                ),
                secondary_y=True,
            )

    if add_dropdown and traces_per_symbol > 0 and len(symbols) > 1:
        total_traces = traces_per_symbol * len(symbols)
        buttons = []
        for idx, sym in enumerate(symbols):
            visibility = [False] * total_traces
            start = idx * traces_per_symbol
            for j in range(traces_per_symbol):
                visibility[start + j] = True
            title = (group.get("title") or f"Series for {sym}").format(symbol=sym)
            buttons.append(
                dict(
                    label=str(sym),
                    method="update",
                    args=[{"visible": visibility}, {"title": title}],
                )
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=1.0,
                    xanchor="right",
                    y=1.15,
                    yanchor="top",
                    showactive=True,
                )
            ]
        )

    primary_title = group.get("primary_title") or (primaries[0][1] if primaries else "Fills")
    secondary_title = group.get("secondary_title") or (secondaries[0][1] if secondaries else "Secondary")

    fig.update_layout(
        title=(group.get("title") or f"Series for {initial_symbol}").format(symbol=initial_symbol),
        hovermode="x unified",
        legend_title="Series",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
        height=650,
    )
    fig.update_xaxes(title_text="Date", type="date")
    fig.update_yaxes(title_text=primary_title, secondary_y=False)
    fig.update_yaxes(title_text=secondary_title, secondary_y=True)
    return fig



__all__ = [
    "create_fake_daily_frame",
    "create_fake_pnl_frame",
    "generate_fake_daily_frames",
    "concat_daily_frames",
    "plot_groups",
    "GroupSpec",
]
