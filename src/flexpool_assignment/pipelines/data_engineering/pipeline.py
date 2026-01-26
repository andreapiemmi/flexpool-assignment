from kedro.pipeline import Pipeline, node
from .nodes import preprocessing, make_supervised_table

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocessing,
                inputs=dict(df="raw_data", dt_col="params:de.dt_col"),
                outputs="preprocessed_data",
                name="preprocess_node",
            ),
            node(
                func=make_supervised_table,
                inputs=dict(df="preprocessed_data", params="params:de"),
                outputs="supervised_table",
                name="features_node",
            ),
        ]
    )
