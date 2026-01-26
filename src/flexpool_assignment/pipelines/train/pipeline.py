from kedro.pipeline import Pipeline, node
from .nodes import (
    split_train_val, 
    train_linear_ols_torch, 
    train_ridge_lasso_enet_torch, 
    train_hgbr_sklearn,
    compare_and_select )


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_train_val,
                inputs=dict(supervised_table="supervised_table", params="params:train"),
                outputs="split_data",
                name="split_node",
            ),
            node(
                train_linear_ols_torch, 
                dict(split_data="split_data", params="params:train"), 
                "ols_artifacts", name="ols"),
            node(train_ridge_lasso_enet_torch, 
                 dict(split_data="split_data", params="params:train"), 
                 "enet_artifacts", name="enet"),
            node(train_hgbr_sklearn, 
                 dict(split_data="split_data", params="params:train"), 
                 "hgbr_artifacts", name="hgbr"),
            node(compare_and_select,
                 dict(
                     ols_artifacts="ols_artifacts",
                    enet_artifacts="enet_artifacts",
                    hgbr_artifacts="hgbr_artifacts"
                 ),["model_artifacts", "metrics_table"], name = 'select')
        ]
    )
