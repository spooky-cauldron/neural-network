use crate::neuron::Neuron;
use crate::value_db::ValueDb;
use crate::ID;

pub struct NeuralNetwork {
    value_db: ValueDb,
    input_layer: Vec<ID>,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        NeuralNetwork { value_db: ValueDb::new(), layers: vec![], input_layer: vec![] }
    }

    pub fn add_layer(&mut self, n_inputs: u64, n_outputs: u64) {
        let layer = Layer::new(n_inputs, n_outputs, &mut self.value_db);
        self.input_layer = layer.input(&self.input_layer, &mut self.value_db);
        self.layers.push(layer);
    }

    pub fn set_input_layer(&mut self, inputs: &[f32]) {
        self.input_layer = inputs.iter()
            .map(|input| self.value_db.push(input.clone()))
            .collect();
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

    pub fn input(&self, input_ids: &[ID], db: &mut ValueDb) -> Vec<ID> {
        let outputs = self.neurons.iter()
            .map(|neuron| neuron.input(input_ids, db))
            .collect();
        return outputs;
    }
}