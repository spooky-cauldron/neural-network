# Neural Network

Simple neural network inspired by [micrograd](https://github.com/karpathy/micrograd) written in Rust.

## Run Locally

```
cargo run
```

For maximum performance, use:

```
cargo run --release
```

## Visualize Network

Saved networks can be viewed using the `visualizer/view_graph.py` Python script.
The visualization shows the weight and gradient of each node in the network along with the operations which generate the final output.

```
cd visualizer
pip install -r requirements.txt
python view_graph.py
```

![example-model](visualizer/example-model.pdf)

## Run Tests

```
cargo test
```
