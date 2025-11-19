"""
Utilities for experimenting with synthetic fill + market data.

The workflow is:
    1. Generate fake dataframes so we can iterate quickly on the plot.
    2. (next step) Feed the dataframes to an interactive chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype

COLOR_CONFIG = {
    "bid": "#1f77b4",   # blue
    "ask": "#d62728",   # red
    "mid": "#4b5563",   # neutral gray
    "fair": "#f97316",  # orange
    "fill": "#4b5563",
}

SESSION_START = "09:30"
SESSION_END = "16:00"
SESSION_START_DELTA = pd.to_timedelta(f"{SESSION_START}:00" if SESSION_START.count(":") == 1 else SESSION_START)
SESSION_END_DELTA = pd.to_timedelta(f"{SESSION_END}:00" if SESSION_END.count(":") == 1 else SESSION_END)


@dataclass(frozen=True)
class FakeDataConfig:
    symbols: Tuple[str, ...] = ("ABC", "DEF")
    fill_events_per_symbol: int = 10
    freq: str = "s"  # pandas offset alias (seconds by default)
    seed: int = 731
    start_date: str = "2024-01-02"  # YYYY-MM-DD
    num_days: int = 2


def concat_feather_by_date(parent_path: str | Path, relative_file: str = "output.fth") -> pd.DataFrame:
    """
    Walk parent_path/YYYYMMDD/relative_file trees, concat all feather tables found.
    *relative_file* can point to fills or market datasets.
    """
    parent = Path(parent_path)
    frames: List[pd.DataFrame] = []
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if len(name) == 8 and name.isdigit():
            fp = child / relative_file
            if fp.exists():
                frames.append(pd.read_feather(fp))
    if not frames:
        raise FileNotFoundError(f"No feather files found under {parent}/*/{relative_file}")
    return pd.concat(frames, ignore_index=True)


def create_fake_dataframes(config: FakeDataConfig | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (fills_df, market_df) matching the schemas requested by the user.
    """
    config = config or FakeDataConfig()
    rng = np.random.default_rng(config.seed)

    base_day = pd.Timestamp(config.start_date)
    timestamps = []
    for day in range(config.num_days):
        day_date = base_day + pd.Timedelta(days=day)
        session_start = day_date + SESSION_START_DELTA
        session_end = day_date + SESSION_END_DELTA
        timestamps.append(pd.date_range(session_start, session_end, freq=config.freq))
    received_times = pd.DatetimeIndex(np.concatenate(timestamps))
    received_times_ns = received_times.view("int64")

    market_df = pd.DataFrame({"ReceivedTime": received_times_ns})
    fills_records: List[dict] = []
    total_points = len(received_times)

    for idx, symbol in enumerate(config.symbols):
        base = 95 + (idx * 10)
        mid_deltas = rng.normal(loc=0, scale=0.05, size=total_points).cumsum()
        mid = base + mid_deltas

        # Fair value follows its own slower path with gentle mean reversion toward base.
        fair_noise = rng.normal(loc=0, scale=0.02, size=total_points)
        fair_mid = np.empty_like(fair_noise)
        fair_mid[0] = base
        alpha = 0.02
        for t in range(1, total_points):
            fair_mid[t] = fair_mid[t - 1] + fair_noise[t] - alpha * (fair_mid[t - 1] - base)

        spread = rng.uniform(0.05, 0.25, size=total_points)

        market_df[f"{symbol}_fair_value_mid_px"] = fair_mid
        market_df[f"{symbol}_bid_px"] = mid - spread / 2
        market_df[f"{symbol}_ask_px"] = mid + spread / 2

        fill_indices = rng.choice(total_points, size=config.fill_events_per_symbol, replace=False)
        fill_indices.sort()
        for fi in fill_indices:
            fills_records.append(
                {
                    "symbol": symbol,
                    "rxTime": int(received_times_ns[fi]),
                    "OrderPrice": float(mid[fi] + rng.normal(0, 0.03)),
                    "FillSize": int(rng.integers(100, 1000, endpoint=False)),
                    "TriggerName": f"{symbol}_strategy_{rng.integers(1,4)}",
                }
            )

    fills_df = pd.DataFrame(fills_records).sort_values("rxTime").reset_index(drop=True)
    return fills_df, market_df


def infer_symbols(market_df: pd.DataFrame) -> List[str]:
    suffix = "_bid_px"
    symbols = sorted(
        col[: -len(suffix)] for col in market_df.columns if col.endswith(suffix) and len(col) > len(suffix)
    )
    return symbols


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return series
    if is_integer_dtype(series):
        return pd.to_datetime(series.astype("int64"), unit="ns")
    return pd.to_datetime(series)


def _filter_triggers(fills_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if "TriggerName" not in fills_df.columns:
        return fills_df
    prefix = f"{symbol}_"
    mask = fills_df["TriggerName"].astype(str).str.startswith(prefix)
    return fills_df[mask]


def _parse_session_time(value: str) -> pd.Timedelta:
    return pd.to_timedelta(value if value.count(":") >= 2 else f"{value}:00")


SESSION_START_DELTA = _parse_session_time(SESSION_START)
SESSION_END_DELTA = _parse_session_time(SESSION_END)


def _hour_fraction(time_str: str) -> float:
    parts = [int(p) for p in time_str.split(":")]
    while len(parts) < 3:
        parts.append(0)
    h, m, s = parts
    return h + m / 60 + s / 3600


def filter_time_window(
    fills_df: pd.DataFrame,
    market_df: pd.DataFrame,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return copies of the data filtered to the inclusive datetime interval.
    *start* and *end* can be strings (YYYY-MM-DD HH:MM:SS) or pandas Timestamps.
    """
    def _coerce(value: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
        if value is None or value == "":
            return None
        if isinstance(value, pd.Timestamp):
            return value
        return pd.to_datetime(value)

    start_ts = _coerce(start)
    end_ts = _coerce(end)

    if start_ts is None and end_ts is None:
        return fills_df.copy(), market_df.copy()

    fills_times = _ensure_datetime(fills_df["rxTime"])
    market_times = _ensure_datetime(market_df["ReceivedTime"])

    def _mask(times: pd.Series) -> pd.Series:
        mask = pd.Series(True, index=times.index)
        if start_ts is not None:
            mask &= times >= start_ts
        if end_ts is not None:
            mask &= times <= end_ts
        return mask

    fills_mask = _mask(fills_times)
    market_mask = _mask(market_times)

    return fills_df[fills_mask].copy(), market_df[market_mask].copy()


class SymbolMarketPlotter:
    """
    Hold fills + market dataframes and provide convenience constructors/plotting helpers.
    """
    def __init__(self, fills_df: pd.DataFrame, market_df: pd.DataFrame) -> None:
        self.fills_df = fills_df.copy()
        self.market_df = market_df.copy()
        self._widget_cache: Any | None = None

    @classmethod
    def from_dated_feathers(
        cls,
        parent_path: str | Path,
        fills_relative: str = "fills/output.fth",
        market_relative: str = "market/output.fth",
    ) -> "SymbolMarketPlotter":
        fills_df = concat_feather_by_date(parent_path, fills_relative)
        market_df = concat_feather_by_date(parent_path, market_relative)
        return cls(fills_df, market_df)

    def time_bounds(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        times = _ensure_datetime(self.market_df["ReceivedTime"])
        return times.min(), times.max()

    def plot(self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None) -> go.Figure:
        filtered_fills, filtered_market = filter_time_window(self.fills_df, self.market_df, start, end)
        return self._build_symbol_dropdown_plot(filtered_fills, filtered_market)

    def time_filter_widget(self) -> Any:
        """
        Return an ipywidgets container with time filtering controls + plot output.
        The widget is cached so calling this repeatedly reuses the same instance.
        """
        if self._widget_cache is not None:
            return self._widget_cache

        try:
            import ipywidgets as widgets
            from IPython.display import display as ipy_display
        except ImportError as exc:  # pragma: no cover - import guard for users without ipywidgets
            raise RuntimeError("ipywidgets and IPython are required for time_filter_widget()") from exc

        start_bound, end_bound = self.time_bounds()
        start_date = widgets.DatePicker(value=start_bound.date(), description="Start date")
        start_time = widgets.Text(value=start_bound.strftime("%H:%M:%S"), description="Start time")
        end_date = widgets.DatePicker(value=end_bound.date(), description="End date")
        end_time = widgets.Text(value=end_bound.strftime("%H:%M:%S"), description="End time")
        output = widgets.Output()

        def _combine(date_widget: widgets.DatePicker, time_widget: widgets.Text) -> Optional[pd.Timestamp]:
            if date_widget.value is None:
                return None
            text = (time_widget.value or "00:00:00").strip()
            return pd.to_datetime(f"{date_widget.value} {text}")

        def render_plot(*_: Any) -> None:
            output.clear_output(wait=True)
            try:
                start_val = _combine(start_date, start_time)
                end_val = _combine(end_date, end_time)
            except ValueError:
                with output:
                    print("Use HH:MM:SS for times.")
                return

            if start_val and end_val and start_val > end_val:
                with output:
                    print("Start must be before end.")
                return

            fig = self.plot(start_val, end_val)
            with output:
                ipy_display(fig)

        for widget in (start_date, start_time, end_date, end_time):
            widget.observe(render_plot, names="value")

        render_plot()
        controls = widgets.VBox(
            [
                widgets.HBox([start_date, start_time]),
                widgets.HBox([end_date, end_time]),
            ]
        )
        self._widget_cache = widgets.VBox([controls, output])
        return self._widget_cache

    def display_time_filter_widget(self) -> None:
        """
        Display (or update) the cached time filter widget. Safe to call repeatedly in notebooks.
        """
        widget = self.time_filter_widget()
        try:
            from IPython.display import display as ipy_display, clear_output
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("IPython is required to display widgets") from exc

        clear_output(wait=True)
        ipy_display(widget)

    @staticmethod
    def _build_symbol_dropdown_plot(fills_df: pd.DataFrame, market_df: pd.DataFrame) -> go.Figure:
        market_df = market_df.sort_values("ReceivedTime").copy()
        fills_df = fills_df.sort_values("rxTime").copy()

        market_df["ReceivedTime"] = _ensure_datetime(market_df["ReceivedTime"])
        fills_df["rxTime"] = _ensure_datetime(fills_df["rxTime"])

        symbols = infer_symbols(market_df)

        fig = go.Figure()
        symbol_trace_map: Dict[str, List[int]] = {}

        for idx, symbol in enumerate(symbols):
            traces_for_symbol: List[int] = []
            bid_col = f"{symbol}_bid_px"
            ask_col = f"{symbol}_ask_px"
            fair_col = f"{symbol}_fair_value_mid_px"
            time_series = market_df["ReceivedTime"]

            traces_for_symbol.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    name=f"{symbol} bid",
                    x=time_series,
                    y=market_df[bid_col],
                    line=dict(color=COLOR_CONFIG["bid"]),
                    visible=(idx == 0),
                )
            )

            traces_for_symbol.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    name=f"{symbol} ask",
                    x=time_series,
                    y=market_df[ask_col],
                    line=dict(color=COLOR_CONFIG["ask"]),
                    visible=(idx == 0),
                )
            )

            mid_values = (market_df[bid_col] + market_df[ask_col]) / 2
            traces_for_symbol.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    name=f"{symbol} mid",
                    x=time_series,
                    y=mid_values,
                    line=dict(color=COLOR_CONFIG["mid"], width=2, dash="dot"),
                    visible=(idx == 0),
                )
            )

            traces_for_symbol.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    name=f"{symbol} fair value",
                    x=time_series,
                    y=market_df[fair_col],
                    line=dict(color=COLOR_CONFIG["fair"], width=2),
                    visible=(idx == 0),
                )
            )

            fills_subset = fills_df[fills_df["symbol"] == symbol]
            fills_subset = _filter_triggers(fills_subset, symbol)
            traces_for_symbol.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    name=f"{symbol} fills",
                    x=fills_subset["rxTime"],
                    y=fills_subset["OrderPrice"],
                    mode="markers",
                    marker=dict(
                        color=COLOR_CONFIG["fill"],
                        size=4 + (fills_subset["FillSize"] / fills_subset["FillSize"].max()) * 12,
                        sizemode="diameter",
                        symbol="x",
                    ),
                    customdata=fills_subset[["FillSize"]],
                    hovertemplate="Time: %{x|%Y-%m-%d %H:%M:%S}<br>Price: %{y:.4f}<br>Shares: %{customdata[0]}<extra></extra>",
                    visible=(idx == 0),
                )
            )

            symbol_trace_map[symbol] = traces_for_symbol

        buttons = []
        total_traces = len(fig.data)
        for symbol in symbols:
            visibility = [False] * total_traces
            for trace_idx in symbol_trace_map[symbol]:
                visibility[trace_idx] = True
            buttons.append(
                dict(
                    label=symbol,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": f"{symbol} market + fills"},
                    ],
                )
            )

        fig.update_layout(
            title=f"{symbols[0]} market + fills" if symbols else "Market + fills",
            xaxis_title="Time",
            yaxis_title="Price",
            legend_title="Legend",
            hovermode="x unified",
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    active=0,
                    x=0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                )
            ],
            legend=dict(
                orientation="h",
                x=0,
                y=-0.2,
                xanchor="left",
                yanchor="top",
            ),
        )
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[_hour_fraction(SESSION_END), _hour_fraction(SESSION_START)], pattern="hour"),
            ]
        )
        return fig


def build_symbol_dropdown_plot(
    fills_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> go.Figure:
    return SymbolMarketPlotter._build_symbol_dropdown_plot(fills_df, market_df)




def main(output_html: str | Path = "symbol_market_plot.html") -> Path:
    fills_df, market_df = create_fake_dataframes()
    plotter = SymbolMarketPlotter(fills_df, market_df)
    fig = plotter.plot()
    output_html = Path(output_html).resolve()
    fig.write_html(output_html, include_plotlyjs="cdn", auto_open=False)
    print(f"Wrote interactive plot to {output_html}")
    return output_html


if __name__ == "__main__":
    main()
