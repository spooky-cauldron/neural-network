use crate::ID;
use crate::value_db::ValueDb;
use rand::prelude::*;

pub struct Neuron {
    weights: Vec<ID>,
    bias: ID,
}

impl Neuron {
    pub fn new(n_inputs: u64, db: &mut ValueDb) -> Self {
        let bias = db.push(random_between(-1.0, 1.0));
        let mut weights = vec![];
        for _ in 0..n_inputs {
            let weight = db.push(random_between(-1.0, 1.0));
            weights.push(weight);
        }
        return Neuron { weights, bias };
    }

    pub fn forward(&self, input_ids: &[ID], db: &mut ValueDb) -> ID {
        let weighted: Vec<ID> = input_ids.into_iter()
            .zip(self.weights.iter())
            .map(|(input_id, weight_id)| db.op_mul(*input_id, *weight_id))
            .collect();
        let weighted_sum = weighted.into_iter()
            .reduce(|acc, e| db.op_add(acc, e))
            .unwrap();
        let activation = db.op_add(weighted_sum, self.bias);
        let output = db.op_tanh(activation);
        return output;
    }

    pub fn parameters(&self) -> Vec<ID> {
        let mut params = self.weights.clone();
        params.push(self.bias);
        return params;
    }
}

fn random_between(min: f32, max: f32) -> f32 {
    min + (max - min) * random::<f32>()
}