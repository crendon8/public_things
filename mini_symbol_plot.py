"""
Reduced multi-day market plotter with dropdown symbol switching.
"""

from __future__ import annotations

from pathlib import Path

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
