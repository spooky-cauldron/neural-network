use neural_network::neural_net::NeuralNetwork;


fn main() {
    let network_shape = [3, 4, 4, 1];
    let mut nn = NeuralNetwork::new(&network_shape);
    let inputs = [2.0, -3.0, 8.0];
    let input_layer = nn.add_values(&inputs);

    nn.forward(input_layer);

    nn.save("./saves/model_0.txt");
}

