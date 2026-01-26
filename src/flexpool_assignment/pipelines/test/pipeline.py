"""
This is a boilerplate pipeline 'test'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa

from .nodes import backtest_winner_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=backtest_winner_model,
                inputs=dict(
                    supervised_table="supervised_table",
                    model_artifacts="model_artifacts",
                    params="params:test",
                ),
                outputs=["backtest_daily", "backtest_hourly"],
                name="backtest",
            )
        ]
    )