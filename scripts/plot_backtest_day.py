import pandas as pd
import plotly.graph_objects as go


df = pd.read_parquet("data/08_reporting/backtest_hourly.parquet")

# Mediam MAE - is a representative day, leaves all other 
# but should flex pool instead ask themselves 
# What's up with the worst day, i.e. how bad things can 
# really get? I leave this to the future eventually :) 


daily_mae = (
    df.groupby("forecast_day", as_index=False)
      .agg(mae=("abs_error", "mean"))
      .sort_values("mae")
      .reset_index(drop=True)
)

chosen_day = daily_mae.iloc[len(daily_mae) // 2]["forecast_day"]

df_day = (
    df[df["forecast_day"] == chosen_day]
    .sort_values("datetime_local")
    .reset_index(drop=True)
)


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_day["datetime_local"],
        y=df_day["consumption_MWh"],
        mode="lines",
        name="Actual",
        line=dict(width=2),
    )
)

fig.add_trace(
    go.Scatter(
        x=df_day["datetime_local"],
        y=df_day["y_pred"],
        mode="lines",
        name="Predicted",
        line=dict(width=2, dash="dash"),
    )
)

fig.update_layout(
    title=f"Next-day load forecast vs actual ({chosen_day})",
    xaxis_title="Hour (local time)",
    yaxis_title="Electricity consumption (MWh)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    margin=dict(l=40, r=40, t=60, b=40),
)

fig.show()
