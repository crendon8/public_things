# public_things

## Symbol plot demo
1. Install the dependencies (once): `pip install pandas numpy plotly ipywidgets`
2. Run `python symbol_market_plot.py` — this generates synthetic fill/market dataframes and writes an interactive Plotly HTML file named `symbol_market_plot.html`.
3. Open the HTML file in your browser. Use the dropdown in the upper-right corner to switch between symbols; each view shows bid/ask/mid lines plus fill markers for the currently selected symbol.

### Notebook workflow
1. Launch Jupyter (`jupyter lab` or `jupyter notebook`) from this directory.
2. Open `symbol_market_plot.ipynb` and execute the cells in order; the data cell calls `create_fake_dataframes()` once and passes the result to `SymbolMarketPlotter(fills_df, market_df)`. The final cell constructs the ipywidgets controls inline (see the notebook) and displays them once per run.

### Loading dated feather files
Use `SymbolMarketPlotter.from_dated_feathers("/somewhere", fills_relative="fills/output.fth", market_relative="market/output.fth")` to instantiate the helper class with your real data. Behind the scenes it calls `concat_feather_by_date` for each dataset so you can immediately call `.plot(...)` or wire the data into custom widgets (see the notebook).
