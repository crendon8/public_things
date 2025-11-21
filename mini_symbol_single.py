
"""
Single-symbol plotter
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype

from mini_fake_data import make_fake_frames

COLOR_MAP = dict(fill="#4b5563", mid="#000000")
# Distinct palettes so bid/ask do not share the same color.
BID_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
ASK_COLORS = ["#d62728", "#ff7f0e", "#e377c2", "#7f7f7f", "#bcbd22"]


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return series
    if is_integer_dtype(series):
        return pd.to_datetime(series.astype("int64"), unit="ns")
    return pd.to_datetime(series)


def concat_feather_by_date(root: str | Path, relative: str = "output.fth") -> pd.DataFrame:
    root = Path(root)
    frames = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.isdigit():
            fp = child / relative
            if fp.exists():
                frames.append(pd.read_feather(fp))
    if not frames:
        raise FileNotFoundError(f"no feather tables under {root}")
    return pd.concat(frames, ignore_index=True)


def plot_single_symbol(
    fills_df: pd.DataFrame,
    market_df: pd.DataFrame,
    symbol: str,
    session_start: Optional[str] = None,
    session_end: Optional[str] = None,
    market_series: Optional[Sequence[dict]] = None,
    tz: str = "America/Chicago",
) -> go.Figure:
    market_df = market_df.sort_values("ReceivedTime").copy()
    fills_df = fills_df.sort_values("rxTime").copy()
    market_df["ReceivedTime"] = _ensure_datetime(market_df["ReceivedTime"])
    fills_df["rxTime"] = _ensure_datetime(fills_df["rxTime"])

    def _convert_to_tz(series: pd.Series) -> pd.Series:
        try:
            return series.dt.tz_localize("UTC").dt.tz_convert(tz)
        except TypeError:
            return series.dt.tz_convert(tz)

    market_df["ReceivedTime"] = _convert_to_tz(market_df["ReceivedTime"])
    fills_df["rxTime"] = _convert_to_tz(fills_df["rxTime"])
    times = market_df["ReceivedTime"]

    def _parse_time(value: Optional[str]) -> Optional[pd.Timedelta]:
        if not value:
            return None
        text = value.replace("::", ":").strip()
        if text.count(":") == 1:
            text = f"{text}:00"
        elif text.count(":") == 0:
            text = f"{text}:00:00"
        return pd.to_timedelta(text)

    start_delta = _parse_time(session_start)
    end_delta = _parse_time(session_end)

    if start_delta and end_delta and start_delta > end_delta:
        raise ValueError("session_start must be before session_end")

    if start_delta is not None:
        market_intraday = market_df["ReceivedTime"] - market_df["ReceivedTime"].dt.normalize()
        fills_intraday = fills_df["rxTime"] - fills_df["rxTime"].dt.normalize()
        market_df = market_df[market_intraday >= start_delta]
        fills_df = fills_df[fills_intraday >= start_delta]

    if end_delta is not None:
        market_intraday = market_df["ReceivedTime"] - market_df["ReceivedTime"].dt.normalize()
        fills_intraday = fills_df["rxTime"] - fills_df["rxTime"].dt.normalize()
        market_df = market_df[market_intraday <= end_delta]
        fills_df = fills_df[fills_intraday <= end_delta]

    if not market_series:
        raise ValueError("market_series is required; provide bid/ask column specs.")

    def _normalize_specs(series: Sequence[dict]) -> list[dict]:
        specs = []
        for spec in series:
            if "bid" not in spec or "ask" not in spec:
                raise ValueError("Each market_series entry must include 'bid' and 'ask' keys")
            label = spec.get("label")
            if not label:
                base = spec["bid"].removesuffix("_bid")
                label = base
            normalized = {"bid": spec["bid"], "ask": spec["ask"], "label": label}
            if "mid" in spec:
                normalized["mid"] = spec["mid"]
            specs.append(normalized)
        return specs

    series_specs = _normalize_specs(market_series)

    fig = go.Figure()
    times = market_df["ReceivedTime"]
    for idx, spec in enumerate(series_specs):
        bid_color = BID_COLORS[idx % len(BID_COLORS)]
        ask_color = ASK_COLORS[idx % len(ASK_COLORS)]
        fig.add_trace(
            go.Scatter(
                name=f"{spec['label']} bid",
                x=times,
                y=market_df[spec["bid"]],
                line=dict(color=bid_color, width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                name=f"{spec['label']} ask",
                x=times,
                y=market_df[spec["ask"]],
                line=dict(color=ask_color, dash="dash"),
            )
        )
        if "mid" in spec:
            fig.add_trace(
                go.Scatter(
                    name=f"{spec['label']} mid",
                    x=times,
                    y=market_df[spec["mid"]],
                    line=dict(color=COLOR_MAP["mid"], dash="dot"),
                )
            )
    fills = fills_df[fills_df["symbol"] == symbol]
    fig.add_trace(
        go.Scatter(
            name=f"{symbol} fills",
            x=fills["rxTime"],
            y=fills["price"],
            mode="markers",
            marker=dict(color=COLOR_MAP["fill"], size=6),
        )
    )
    fig.update_layout(title=f"{symbol} market + fills", hovermode="x unified")
    return fig


def main(
    symbol: str = "ABC",
    start: Optional[str] = None,
    end: Optional[str] = None,
    output: Optional[str] = None,
    market_series: Optional[Sequence[dict]] = None,
    tz: str = "America/Chicago",
) -> Path:
    fills_df, market_df = make_fake_frames()
    if market_series is None:
        raise ValueError("market_series must be provided when running main()")
    fig = plot_single_symbol(fills_df, market_df, symbol, start, end, market_series=market_series, tz=tz)
    if output is None:
        fig.show()
        return Path(".")
    output_path = Path(output).resolve()
    fig.write_html(output_path, include_plotlyjs="cdn", auto_open=False)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a single symbol market plot.")
    parser.add_argument("--symbol", default="ABC", help="Symbol to display (default: ABC)")
    parser.add_argument("--start", help='Session start time, e.g. "09:30" or "09:30:00"')
    parser.add_argument("--end", help='Session end time, e.g. "16:00" or "16:00:00"')
    parser.add_argument("--output", help="Optional HTML output path; if omitted fig.show() is used.")
    parser.add_argument(
        "--market-series",
        required=True,
        help="JSON string describing bid/ask columns, e.g. "
        "'[{\"label\":\"NYSE px0\",\"bid\":\"NYSE.ABC_bid\",\"ask\":\"NYSE.ABC_ask\"}]'",
    )
    parser.add_argument("--tz", default="America/Chicago", help="Timezone for interpreting session start/end (default: America/Chicago)")
    args = parser.parse_args()
    import json

    market_series = json.loads(args.market_series)
    path = main(args.symbol, args.start, args.end, args.output, market_series=market_series, tz=args.tz)
    if args.output:
        print(f"Wrote plot to {path}")
