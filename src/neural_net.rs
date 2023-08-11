use crate::neuron::Neuron;
use crate::value_db::ValueDb;
use crate::ID;
use crate::value::Value;

pub struct NeuralNetwork {
    value_db: ValueDb,
    layers: Vec<Layer>,
    pub core_value_count: usize,
}

impl NeuralNetwork {
    pub fn new_empty() -> Self {
        NeuralNetwork { value_db: ValueDb::new(), layers: vec![], core_value_count: 0 }
    }

    pub fn new(layer_sizes: &[u64]) -> Self {
        let mut nn = NeuralNetwork::new_empty();
        for i in 1..layer_sizes.len() {
            let n_inputs = layer_sizes[i - 1];
            let n_outputs = layer_sizes[i];
            nn.add_layer(n_inputs, n_outputs)
        }
        nn.core_value_count = nn.value_db.len();
        return nn;
    }

    pub fn add_layer(&mut self, n_inputs: u64, n_outputs: u64) {
        let layer = Layer::new(n_inputs, n_outputs, &mut self.value_db);
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input_data_ids: Vec<ID>) -> Vec<ID> {
        let mut active_layer = input_data_ids;
        for i in 0..self.layers.len() {
            active_layer = self.layers[i].forward(&active_layer, &mut self.value_db);
        }
        return active_layer;
    }

    pub fn reset(&mut self) {
        if self.core_value_count == 0 {
            return;
        }
        self.value_db.clear(self.core_value_count + 1);
    }

    pub fn backward(&mut self, from: ID) {
        self.value_db.backward(from);
    }

    pub fn zero_grad(&mut self) {
        self.value_db.zero_grad();
    }

    pub fn parameters(&self) -> Vec<ID> {
        self.layers.iter()
            .map(|layer| layer.parameters())
            .collect::<Vec<Vec<ID>>>()
            .concat()
    }

    pub fn optimize(&mut self, rate: f32) {
        for param_id in self.parameters() {
            let param = self.value_db.get_mut(param_id);
            param.value += param.grad * rate * -1.0;
        }
    }

    pub fn add_values(&mut self, inputs: &[f32]) -> Vec<ID> {
        let added_value_ids = inputs.iter()
            .map(|input| self.value_db.push(input.clone()))
            .collect();
        return added_value_ids;
    }

    pub fn get_value(&mut self, id: ID) -> &Value {
        self.value_db.get(id)
    }

    pub fn loss(&mut self, prediction_ids: Vec<ID>, label_ids: Vec<ID>) -> ID {
        let mut losses = vec![];
        for (prediction_id, label_id) in prediction_ids.iter().zip(label_ids) {
            let negate = self.value_db.push(-1.0);
            let negative_prediction = self.value_db.op_mul(*prediction_id, negate);
            let diff = self.value_db.op_add(label_id, negative_prediction);
            let diff_squared = self.value_db.op_mul(diff, diff);
            losses.push(diff_squared);
        }
        let total_loss_id = losses.into_iter().reduce(|acc, e| self.value_db.op_add(acc, e));
        return total_loss_id.unwrap();
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

    pub fn parameters(&self) -> Vec<ID> {
        self.neurons.iter()
            .map(|neuron| neuron.parameters())
            .collect::<Vec<Vec<ID>>>()
            .concat()
    }
}