"""
This is a boilerplate pipeline 'train'
generated using Kedro 1.0.0
"""
from __future__ import annotations
from kedro.pipeline import Pipeline, node, pipeline

import flexpool_assignment.common.ml_utils as ml_utils
from .nodes import train_torch_linear, predict_baseline_torch_linear


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = ml_utils.fit_scaler, 
            inputs = "X_train",
            outputs="feature_scaler",
            name = "fit_feature_scaler_node",
            tags = ["preprocess"]
        ),
        node(
            func = train_torch_linear, 
            inputs = [
                "X_train",
                "Y_train", 
                "feature_scaler",
                "params:train.seed",
                "params:train.batch_size",
                "params:train.epochs",
                "params:train.lr",
                "params:train.weight_decay"
            ], 
            outputs="torch_linear_state",
            name = "train_torch_linear", 
            tags = ["train"]
        ), 
        node(
            func = predict_baseline_torch_linear,
            inputs=["X_train","Y_train", "X_val", "Y_val",
                    "X_test","Y_test","feature_scaler",
                    "torch_linear_state"], 
            outputs=["preds_val", "preds_test"],
            name = "predict_baseline_torch_linear",
            tags=["predict"]
        )

    ])

