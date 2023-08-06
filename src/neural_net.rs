use crate::neuron::Neuron;
use crate::value_db::ValueDb;
use crate::ID;

pub struct NeuralNetwork {
    value_db: ValueDb,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new_empty() -> Self {
        NeuralNetwork { value_db: ValueDb::new(), layers: vec![] }
    }

    pub fn new(layer_sizes: &[u64]) -> Self {
        let mut nn = NeuralNetwork::new_empty();
        for i in 1..layer_sizes.len() {
            let n_inputs = layer_sizes[i - 1];
            let n_outputs = layer_sizes[i];
            nn.add_layer(n_inputs, n_outputs)
        }
        return nn;
    }

    pub fn add_layer(&mut self, n_inputs: u64, n_outputs: u64) {
        let layer = Layer::new(n_inputs, n_outputs, &mut self.value_db);
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input_data_ids: Vec<ID>) {
        let mut input_layer = input_data_ids;
        for i in 0..self.layers.len() {
            input_layer = self.layers[i].forward(&input_layer, &mut self.value_db);
        }
    }

    pub fn add_values(&mut self, inputs: &[f32]) -> Vec<ID> {
        let added_value_ids = inputs.iter()
            .map(|input| self.value_db.push(input.clone()))
            .collect();
        return added_value_ids;
    }

    pub fn save(&self, path: &str) {
        self.value_db.save(path);
    }
}

pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(n_inputs: u64, n_outputs: u64, db: &mut ValueDb)  -> Self {
        let mut neurons = vec![];
        for _ in 0..n_outputs {
            neurons.push(Neuron::new(n_inputs, db));
        }
        return Layer { neurons };
    }

    pub fn forward(&self, input_ids: &[ID], db: &mut ValueDb) -> Vec<ID> {
        let outputs = self.neurons.iter()
            .map(|neuron| neuron.forward(input_ids, db))
            .collect();
        return outputs;
    }
}