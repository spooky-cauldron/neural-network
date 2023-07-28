use std::fs;

#[derive(Debug)]
struct Value {
    pub value: f32,
    grad: f32,
    pub op: Op,
    child_0_id: Option<ID>,
    child_1_id: Option<ID>,
}

impl Value {
    fn new(value: f32) -> Self {
        Value { value, grad: 0.0, op: Op::None , child_0_id: None, child_1_id: None}
    }

    pub fn to_save(&self) -> String {
        let child_ids = [self.child_0_id, self.child_1_id];
        let child_save_ids: Vec<String> = child_ids.iter()
            .map(|child_id| match child_id {
                Some(id) => id.to_string(),
                None => "N".to_owned(),
            })
            .collect();

        format!("{},{},{},{}", 
            self.value.to_string(),
            self.op as u8,
            child_save_ids[0],
            child_save_ids[1],
        ) 
    }
}


#[derive(Debug, Clone, Copy)]
enum Op {
    None,
    Add,
    Multiply,
}

struct ValueDb {
    values: Vec<Value>,
}

type ID = usize;

impl ValueDb {
    fn new() -> Self { ValueDb { values: vec![] } }

    pub fn push(&mut self, value: f32) -> ID {
        self.values.push(Value::new(value));
        return self.values.len() - 1;
    }

    fn join(&mut self, op: Op, parent: ID, children: (Option<ID>, Option<ID>)) {
        self.values[parent].op = op;
        self.values[parent].child_0_id = children.0;
        self.values[parent].child_1_id = children.1;
    }

    pub fn get(&self, id: ID) -> &Value {
        &self.values[id]
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


fn main() {
    let mut db = ValueDb::new();

    let a = db.push(1.0);
    let b = db.push(2.0);
    let c = db.op_add(a, b);

    let d = db.push(4.0);
    let e = db.op_mul(c, d);

    println!("{:?}", db.get(c));
    println!("{:?}", db.get(e));

    db.zero_grad();

    db.save("./saves/model_0.txt")
}

