"""
Reduced multi-day market plotter with dropdown symbol switching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.graph_objects as go

from mini_fake_data import SESSION_END, SESSION_START, make_fake_frames

COLOR_MAP = dict(bid="#1f77b4", ask="#d62728", fill="#4b5563", mid="#000000")


def concat_feather_by_date(root: str | Path, relative: str = "output.fth") -> pd.DataFrame:
    root = Path(root)
    frames = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.isdigit():
            fp = child / relative
            if fp.exists():
                frames.append(pd.read_feather(fp))
    if not frames:
        raise FileNotFoundError(f"no feather tables in {root}/*/{relative}")
    return pd.concat(frames, ignore_index=True)


def _symbols(market: pd.DataFrame) -> list[str]:
    return sorted(col[:-4] for col in market.columns if col.endswith("_bid"))


def _hour_fraction(text: str) -> float:
    parts = (text.split(":") + ["0", "0"])[:3]
    h, m, s = map(int, parts)
    return h + m / 60 + s / 3600


def attach_buy_sell(
    fills_df: pd.DataFrame,
    market_df: pd.DataFrame,
    *,
    prefix: str = "",
    buy_suffix: str = "_bid",
    sell_suffix: str = "_ask",
    fill_time_col: str = "rxTime",
    market_time_col: str = "ReceivedTime",
    symbol_col: str = "symbol",
    buy_col_name: str = "buy",
    sell_col_name: str = "sell",
    time_diff_col_name: str = "time_diff",
    mid_suffix: str = "_mid",
    mid_col_name: str = "mid",
    fill_prem_col_name: str = "fill_prem",
    order_price_col: str | None = None,
    symbol_mask: Sequence[str] | None = ("ABC", "DEF"),
) -> pd.DataFrame:
    """
    Align fills_df and market_df on timestamp + symbol (nearest neighbor) and pull sided prices into fills_df.

    Time columns are coerced with pd.to_datetime so they can be ns integers or datetimes.
    Market columns are expected to follow the pattern "{prefix}{symbol}{buy_suffix}" and
    "{prefix}{symbol}{sell_suffix}" and "{prefix}{symbol}{mid_suffix}". The nearest market timestamp is used
    per fill, the absolute time delta (ns) is stored in *time_diff_col_name*, and the midpoint is used to
    compute a fill premium (bps): (OrderPrice / mid - 1) * 1e4.
    """
    fills = fills_df.copy()
    fills["_time_key"] = pd.to_datetime(fills[fill_time_col])
    fills["_fill_idx"] = fills.index
    fills["_fill_pos"] = range(len(fills))  # positional index to avoid duplicate label issues
    for col in (buy_col_name, sell_col_name, time_diff_col_name, mid_col_name, fill_prem_col_name):
        if col not in fills.columns:
            fills[col] = pd.NA

    if order_price_col is None:
        if "OrderPrice" in fills.columns:
            order_price_col = "OrderPrice"
        elif "price" in fills.columns:
            order_price_col = "price"
        else:
            raise ValueError("order_price_col not provided and neither 'OrderPrice' nor 'price' found in fills_df.")
    elif order_price_col not in fills.columns:
        raise ValueError(f"order_price_col '{order_price_col}' not found in fills_df.")

    market = market_df.copy()
    market["_time_key"] = pd.to_datetime(market[market_time_col])

    symbols = pd.unique(fills[symbol_col]).tolist()
    if symbol_mask is not None:
        symbols = [sym for sym in symbols if sym in symbol_mask]
    missing = []
    for sym in symbols:
        buy_col = f"{prefix}{sym}{buy_suffix}"
        sell_col = f"{prefix}{sym}{sell_suffix}"
        mid_col = f"{prefix}{sym}{mid_suffix}"
        for col in (buy_col, sell_col, mid_col):
            if col not in market.columns:
                missing.append(col)
    if missing:
        raise ValueError(f"Missing columns in market_df: {missing}")

    for sym in symbols:
        buy_col = f"{prefix}{sym}{buy_suffix}"
        sell_col = f"{prefix}{sym}{sell_suffix}"
        mid_col = f"{prefix}{sym}{mid_suffix}"
        mask = fills[symbol_col] == sym
        fill_positions = fills.loc[mask, "_fill_pos"].to_numpy()
        fill_indices = fills.loc[mask, "_fill_idx"]
        market_sym = (
            market[["_time_key", buy_col, sell_col, mid_col]]
            .rename(columns={"_time_key": "_market_time"})
            .sort_values("_market_time")
        )
        fills_sym = fills.loc[mask, ["_fill_idx", "_time_key"]].sort_values("_time_key")
        aligned = pd.merge_asof(
            fills_sym,
            market_sym,
            left_on="_time_key",
            right_on="_market_time",
            direction="nearest",
        )
        delta = (aligned["_market_time"] - aligned["_time_key"]).abs()
        aligned[time_diff_col_name] = delta.astype("int64")  # nanoseconds
        aligned = aligned.set_index("_fill_idx").reindex(fill_indices)
        mid_values = aligned[mid_col].to_numpy()
        fills.iloc[fill_positions, fills.columns.get_loc(buy_col_name)] = aligned[buy_col].to_numpy()
        fills.iloc[fill_positions, fills.columns.get_loc(sell_col_name)] = aligned[sell_col].to_numpy()
        fills.iloc[fill_positions, fills.columns.get_loc(mid_col_name)] = mid_values
        price_values = pd.to_numeric(fills.iloc[fill_positions][order_price_col], errors="coerce").to_numpy()
        prem = (price_values / mid_values - 1.0) * 1e4
        fills.iloc[fill_positions, fills.columns.get_loc(fill_prem_col_name)] = prem
        fills.iloc[fill_positions, fills.columns.get_loc(time_diff_col_name)] = aligned[time_diff_col_name].to_numpy()

    return fills.drop(columns=["_time_key", "_fill_idx", "_fill_pos"])


def plot_market(fills_df: pd.DataFrame, market_df: pd.DataFrame) -> go.Figure:
    market_df = market_df.sort_values("ReceivedTime").copy()
    fills_df = fills_df.sort_values("rxTime").copy()
    symbols = _symbols(market_df)
    fig = go.Figure()
    times = market_df["ReceivedTime"]
    for idx, sym in enumerate(symbols):
        bid = market_df[f"{sym}_bid"]
        ask = market_df[f"{sym}_ask"]
        fig.add_trace(go.Scatter(name=f"{sym} bid", x=times, y=bid, line=dict(color=COLOR_MAP["bid"]), visible=(idx == 0)))
        fig.add_trace(go.Scatter(name=f"{sym} ask", x=times, y=ask, line=dict(color=COLOR_MAP["ask"]), visible=(idx == 0)))
        fills = fills_df[fills_df["symbol"] == sym]
        fig.add_trace(
            go.Scatter(
                name=f"{sym} fills",
                x=fills["rxTime"],
                y=fills["price"],
                mode="markers",
                marker=dict(color=COLOR_MAP["fill"], size=6),
                visible=(idx == 0),
            )
        )
    buttons = []
    total = len(fig.data)
    for idx, sym in enumerate(symbols):
        vis = [False] * total
        start = idx * 3
        for offset in range(3):
            if start + offset < total:
                vis[start + offset] = True
        buttons.append(dict(label=sym, method="update", args=[{"visible": vis}, {"title": f"{sym} market + fills"}]))
    fig.update_layout(
        title=f"{symbols[0]} market + fills" if symbols else "market",
        hovermode="x unified",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=0, xanchor="left", y=1.2)],
    )
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[_hour_fraction(SESSION_END), _hour_fraction(SESSION_START)], pattern="hour"),
        ]
    )
    return fig


def main() -> None:
    fills_df, market_df = make_fake_frames()
    plot_market(fills_df, market_df).show()


if __name__ == "__main__":
    main()
