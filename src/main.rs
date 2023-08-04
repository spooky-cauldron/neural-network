use neural_network::neural_net::NeuralNetwork;


fn main() {
    let mut nn = NeuralNetwork::new();
    let inputs = [2.0, -3.0, 8.0];
    nn.set_input_layer(&inputs);
    nn.add_layer(3, 4);
    nn.add_layer(4, 4);
    nn.add_layer(4, 1);

    nn.save("./saves/model_0.txt");
}

