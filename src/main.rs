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
    
    let mut predictions = vec![];
    for input in dataset {
        let input_layer = nn.add_values(&input);
        let prediction = nn.forward(input_layer);
        predictions.push(prediction[0]);
    }
    let label_ids = nn.add_values(&labels);

    let loss_id = nn.loss(predictions, label_ids);
    let loss = nn.get_value(loss_id).value; 
    println!("Loss: {}", loss);

    nn.backward(loss_id);
        
    nn.save("./saves/model_0.txt");
}

