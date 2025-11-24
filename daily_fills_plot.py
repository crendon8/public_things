"""
Utilities for generating fake daily fills data and plotting them with Plotly.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

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

    # Further break down into buy/sell with small noise and handle pct diffs safely.
    sim_buy_fills = np.round(sim_fills * rng.uniform(0.4, 0.6, size=len(names))).astype(int)
    sim_sell_fills = (sim_fills - sim_buy_fills).astype(int)
    prod_buy_fills = np.round(sim_buy_fills + rng.normal(0, 5, size=len(names))).astype(int)
    prod_sell_fills = np.round(sim_sell_fills + rng.normal(0, 5, size=len(names))).astype(int)
    # Counts-related metrics
    sim_buys = rng.integers(40, 90, size=len(names))
    sim_sells = rng.integers(40, 90, size=len(names))
    prod_buys = sim_buys + rng.normal(0, 8, size=len(names))
    prod_sells = sim_sells + rng.normal(0, 8, size=len(names))
    # Price-based metrics
    sim_buy_px = rng.normal(100, 5, size=len(names))
    sim_sell_px = rng.normal(101, 5, size=len(names))
    prod_buy_px = sim_buy_px + rng.normal(0, 0.5, size=len(names))
    prod_sell_px = sim_sell_px + rng.normal(0, 0.5, size=len(names))
    with np.errstate(divide="ignore", invalid="ignore"):
        total_buy_pct_diff = np.where(
            sim_buy_fills == 0, np.nan, (prod_buy_fills - sim_buy_fills) / sim_buy_fills * 100
        )
        total_sell_pct_diff = np.where(
            sim_sell_fills == 0, np.nan, (prod_sell_fills - sim_sell_fills) / sim_sell_fills * 100
        )
        ct_buy_pct = np.where(sim_buys == 0, np.nan, (prod_buys - sim_buys) / sim_buys * 100)
        ct_sell_ct = np.where(sim_sells == 0, np.nan, (prod_sells - sim_sells) / sim_sells * 100)
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

    Each frame has index=symbol and columns for totals, buy/sell breakdown, count-based metrics, and price-based metrics.
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


def plot_fills_for(df: pd.DataFrame, symbol: str, group_spec: GroupSpec) -> go.Figure:
    """
    Plot a single symbol/group combination (no dropdowns).
    """
    subset = df[df["symbol"] == symbol].sort_values("date")
    if subset.empty:
        raise ValueError(f"No rows found for symbol {symbol}")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    primaries = list(group_spec.get("primary", []))
    secondaries = list(group_spec.get("secondary", []))
    for col, label in primaries:
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset[col],
                name=label,
                mode="lines+markers",
            )
        )
    for col, label in secondaries:
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset[col],
                name=label,
                mode="lines+markers",
            ),
            secondary_y=True,
        )
    fig.update_layout(
        title=(group_spec.get("title") or f"Series for {symbol}").format(symbol=symbol),
        hovermode="x unified",
        legend_title="Series",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
        height=650,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Fills", secondary_y=False)
    fig.update_yaxes(title_text="pct_diff (%)", secondary_y=True)
    return fig


def plot_fills_widget(
    df: pd.DataFrame,
    groups: Mapping[str, GroupSpec] | Sequence[tuple[str, GroupSpec]] | GroupSpec,
    initial_symbol: str | None = None,
    initial_group: str | None = None,
) -> "ipywidgets.VBox":
    """
    Build an ipywidgets UI with two dropdowns (symbol, group) controlling the plot.
    """
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError("ipywidgets is required for plot_fills_widget") from exc

    if isinstance(groups, Mapping):
        group_items = list(groups.items())
    elif isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        if len(groups) == 0:
            raise ValueError("groups must not be empty")
        group_items = [(k, v) if isinstance(k, str) else (str(k), v) for k, v in groups]
    else:
        group_items = [("group", groups)]

    group_lookup = {name: spec for name, spec in group_items}

    symbols = sorted(df["symbol"].unique())
    if not symbols:
        raise ValueError("No symbols found in dataframe")
    symbol_default = initial_symbol if initial_symbol in symbols else symbols[0]
    group_default = initial_group if initial_group in group_lookup else group_items[0][0]

    symbol_dd = widgets.Dropdown(options=symbols, value=symbol_default, description="Symbol")
    group_dd = widgets.Dropdown(options=list(group_lookup.keys()), value=group_default, description="Group")

    base_fig = plot_fills_for(df, symbol_dd.value, group_lookup[group_dd.value])
    fig_widget = go.FigureWidget()
    fig_widget.add_traces(base_fig.data)
    fig_widget.layout = base_fig.layout

    def _update_fig(_=None):
        spec = group_lookup[group_dd.value]
        new_fig = plot_fills_for(df, symbol_dd.value, spec)
        with fig_widget.batch_update():
            fig_widget.data = ()
            fig_widget.add_traces(new_fig.data)
            fig_widget.layout = new_fig.layout

    symbol_dd.observe(_update_fig, names="value")
    group_dd.observe(_update_fig, names="value")

    return widgets.VBox([widgets.HBox([symbol_dd, group_dd]), fig_widget])


def plot_fills(
    df: pd.DataFrame,
    groups: Mapping[str, GroupSpec] | Sequence[tuple[str, GroupSpec]] | GroupSpec,
    add_dropdown: bool = True,
) -> go.Figure:
    """
    Plot series with dropdowns to toggle symbol/group combinations.

    Each group spec should provide "primary" and/or "secondary" lists of (column, label) tuples
    and may include a "title" string (supports {symbol} formatting).
    """
    symbols = sorted(df["symbol"].unique())
    if not symbols:
        raise ValueError("No symbols found in dataframe")

    if isinstance(groups, Mapping):
        group_items = list(groups.items())
    elif isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        if len(groups) == 0:
            raise ValueError("groups must not be empty")
        # Sequence of tuples expected
        group_items = [(k, v) if isinstance(k, str) else (str(k), v) for k, v in groups]
    else:
        # Single group spec passed
        group_items = [("group", groups)]

    initial_symbol = symbols[0]
    initial_group_name, initial_group_spec = group_items[0]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    primaries = [list(spec.get("primary", [])) for _, spec in group_items]
    secondaries = [list(spec.get("secondary", [])) for _, spec in group_items]

    traces_per_combo = [len(p) + len(s) for p, s in zip(primaries, secondaries)]
    if any(tp == 0 for tp in traces_per_combo):
        raise ValueError("Each group must include at least one primary or secondary series")

    visibility_masks: list[list[bool]] = []
    total_traces = sum(tp * len(symbols) for tp in traces_per_combo)
    trace_offset = 0
    combo_offsets: dict[tuple[int, int], int] = {}

    for group_idx, (group_name, spec) in enumerate(group_items):
        prim = primaries[group_idx]
        sec = secondaries[group_idx]
        traces_per_symbol = len(prim) + len(sec)
        for sym_idx, sym in enumerate(symbols):
            combo_offsets[(group_idx, sym_idx)] = trace_offset
            subset = df[df["symbol"] == sym].sort_values("date")
            visible = group_idx == 0 and sym_idx == 0
            for col, label in prim:
                fig.add_trace(
                    go.Scatter(
                        x=subset["date"],
                        y=subset[col],
                        name=f"{label} ({sym})",
                        mode="lines+markers",
                        visible=visible,
                    )
                )
                trace_offset += 1
            for col, label in sec:
                fig.add_trace(
                    go.Scatter(
                        x=subset["date"],
                        y=subset[col],
                        name=f"{label} ({sym})",
                        mode="lines+markers",
                        visible=visible,
                    ),
                    secondary_y=True,
                )
                trace_offset += 1

    def combo_visibility(group_idx: int, sym_idx: int) -> list[bool]:
        mask = [False] * total_traces
        offset = combo_offsets[(group_idx, sym_idx)]
        count = traces_per_combo[group_idx]
        for j in range(count):
            mask[offset + j] = True
        return mask

    if add_dropdown:
        symbol_buttons = []
        group_buttons = []
        for s_idx, sym in enumerate(symbols):
            vis = combo_visibility(0, s_idx)
            title = (group_items[0][1].get("title") or f"Series for {sym}").format(symbol=sym)
            symbol_buttons.append(
                dict(
                    label=str(sym),
                    method="update",
                    args=[{"visible": vis}, {"title": title}, {"meta": {"symbol_idx": s_idx}}],
                )
            )
        for g_idx, (group_name, spec) in enumerate(group_items):
            vis = combo_visibility(g_idx, 0)
            title = (spec.get("title") or f"{group_name} for {symbols[0]}").format(symbol=symbols[0])
            group_buttons.append(
                dict(
                    label=str(group_name),
                    method="update",
                    args=[{"visible": vis}, {"title": title}, {"meta": {"group_idx": g_idx}}],
                )
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=symbol_buttons,
                    direction="down",
                    x=0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    showactive=True,
                ),
                dict(
                    buttons=group_buttons,
                    direction="down",
                    x=0.2,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    showactive=True,
                ),
            ]
        )

    fig.update_layout(
        title=(initial_group_spec.get("title") or f"Series for {initial_symbol}").format(symbol=initial_symbol),
        hovermode="x unified",
        legend_title="Series",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
        height=650,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Fills", secondary_y=False)
    fig.update_yaxes(title_text="pct_diff (%)", secondary_y=True)
    return fig


__all__ = [
    "create_fake_daily_frame",
    "generate_fake_daily_frames",
    "concat_daily_frames",
    "plot_fills",
    "plot_fills_for",
    "plot_fills_widget",
    "GroupSpec",
]
