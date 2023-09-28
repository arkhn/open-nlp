import typer
from datasets import load_dataset
from prompt import promptify

app = typer.Typer()


@app.command()
def create_few_shots_examples(name: str = "fr.layer1") -> None:
    """Create few-shot examples for a given layer.

    This method is used to create few-shot examples in a prompt format. The aim is to use this
    examples to guide a large language model in an extract task.

    Args:
        name: Name of the layer. Defaults to "fr.layer1".
        concat_examples: Number of examples to concatenate. Defaults to 1.
    """
    e3c_dataset = load_dataset("bio-datasets/e3c")
    for example in promptify(
        e3c_dataset[name], e3c_dataset[name].features["clinical_entity_tags"].feature.int2str
    ):
        print(example)


if __name__ == "__main__":
    app()
