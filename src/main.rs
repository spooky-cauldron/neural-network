use neural_network::neural_net::NeuralNetwork;


fn main() {
    let dataset = [
        [2.0, -3.0, 8.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ];
    let labels = [1.0, -1.0, -1.0, 1.0];

    let network_shape = [3, 4, 4, 1];
    let mut nn = NeuralNetwork::new(&network_shape);

    for input in dataset {
        let input_layer = nn.add_values(&input);
        let prediction = nn.forward(input_layer);
        println!("{:?}", prediction);
    }
    // let inputs = [2.0, -3.0, 8.0];

    // nn.forward(input_layer);

    nn.save("./saves/model_0.txt");
}

