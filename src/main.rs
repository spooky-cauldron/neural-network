use neural_network::neural_net::NeuralNetwork;

fn main() {
    let dataset = [
        [2.0, -3.0, 2.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ];
    let labels = [1.0, -1.0, -1.0, 1.0];

    let network_shape = [3, 4, 4, 1];
    let mut nn = NeuralNetwork::new(&network_shape);
    println!("Network Parameter Count: {}", nn.parameters().len());

    let n_epochs = 10000;

    for i in 0..n_epochs {
        nn.reset();
        let mut predictions = vec![];
        for input in dataset {
            let input_layer = nn.add_values(&input);
            let prediction = nn.forward(input_layer);
            predictions.push(prediction[0]);
        }
        let label_ids = nn.add_values(&labels);

        let loss_id = nn.loss(predictions, label_ids);
        let loss = nn.get_value(loss_id).value;
        if i % 100 == 0 {
            println!("Loss: {}", loss);
        }

        nn.zero_grad();
        nn.backward(loss_id);

        let learing_rate = 0.005;
        nn.optimize(learing_rate);
    }

    // eval
    let mut predictions = vec![];
    for input in dataset {
        let input_layer = nn.add_values(&input);
        let prediction = nn.forward(input_layer);
        predictions.push(prediction[0]);
    }
    dbg!(&predictions);
    let results: Vec<f32> = predictions
        .into_iter()
        .map(|id| nn.get_value(id).value)
        .collect();
    dbg!(results);

    // nn.save("./saves/model_0.txt");
}
