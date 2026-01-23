from kedro.pipeline import Pipeline, node, pipeline
from .nodes import hello

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(func=hello, inputs=None, outputs="hello_text", name="hello_node"),
        ]
    )
