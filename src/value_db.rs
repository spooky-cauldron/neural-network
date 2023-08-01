use std::fs;
use std::collections::VecDeque;
use crate::value::*;
use crate::ID;

pub struct ValueDb {
    values: Vec<Value>,
}


impl ValueDb {
    pub fn new() -> Self { ValueDb { values: vec![] } }

    pub fn push(&mut self, value: f32) -> ID {
        self.values.push(Value::new(value));
        return self.values.len() - 1;
    }

    fn join(&mut self, op: Op, parent: ID, children: (Option<ID>, Option<ID>)) {
        self.get_mut(parent).op = op;
        self.get_mut(parent).child_0_id = children.0;
        self.get_mut(parent).child_1_id = children.1;
    }

    pub fn get(&self, id: ID) -> &Value {
        &self.values[id]
    }

    pub fn get_mut(&mut self, id: ID) -> &mut Value {
        &mut self.values[id]
    }

    pub fn backward(&mut self, from: ID) {
        self.get_mut(from).grad = 1.0;
        let mut child_gradients_to_calculate = VecDeque::new();
        child_gradients_to_calculate.push_back(from);

        while child_gradients_to_calculate.len() > 0 {
            let value_id = child_gradients_to_calculate[0];
            self.calculate_child_gradients(value_id);
            let additional_gradients_to_calculate = self.get(value_id).children();
            child_gradients_to_calculate.extend(additional_gradients_to_calculate.iter());
            child_gradients_to_calculate.pop_front();
        }
    }

    fn calculate_child_gradients(&mut self, parent_id: ID) {
        let parent = self.get(parent_id);
        let parent_op = parent.op;
        let parent_grad = parent.grad;
        let children = parent.children();

        match parent_op {
            Op::Add => {
                self.get_mut(children[0]).grad = parent_grad;
                self.get_mut(children[1]).grad = parent_grad;
            }

            Op::Multiply => {
                let child_1_value = self.get(children[1]).value;
                self.get_mut(children[0]).grad = child_1_value * parent_grad;
                let child_0_value = self.get(children[0]).value;
                self.get_mut(children[1]).grad = child_0_value * parent_grad;
            }

            _ => ()
        }
    }

    pub fn save(&self, path: &str) {
        let data: Vec<String> = self.values.iter()
            .map(|value| value.to_save())
            .collect();
        let data_output = data.join(",");
        match fs::write(path, data_output) {
            Ok(_) => println!("Saved model to path: {}", path),
            Err(err) => println!("Error saving: {}", err),
        }
    }
}

impl ValueDb {
    pub fn op_add(&mut self, a: ID, b: ID) -> ID {
        let a_value = self.values[a].value;
        let b_value = self.values[b].value;
        let result = self.push(a_value + b_value);

        self.join(Op::Add, result, (Some(a), Some(b)));

        return result;
    }

    pub fn op_mul(&mut self, a: ID, b: ID) -> ID {
        let a_value = self.values[a].value;
        let b_value = self.values[b].value;
        let result = self.push(a_value * b_value);

        self.join(Op::Multiply, result, (Some(a), Some(b)));

        return result;
    }

    pub fn zero_grad(&mut self) {
        for value in self.values.iter_mut() {
            value.grad = 0.0;
        }
    }
}