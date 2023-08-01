use crate::ID;

#[derive(Debug)]
pub struct Value {
    pub value: f32,
    pub grad: f32,
    pub op: Op,
    pub child_0_id: Option<ID>,
    pub child_1_id: Option<ID>,
}

impl Value {
    pub fn new(value: f32) -> Self {
        Value { value, grad: 0.0, op: Op::None , child_0_id: None, child_1_id: None}
    }

    pub fn children(&self) -> Vec<ID> {
        let child_ids = self.child_slots();
        return child_ids.iter()
            .filter_map(|child| child.clone())
            .collect();
    }

    fn child_slots(&self) -> [Option<ID>; 2] {
        [self.child_0_id, self.child_1_id]
    }

    pub fn to_save(&self) -> String {
        let child_ids = self.child_slots();
        let child_save_ids: Vec<String> = child_ids.iter()
            .map(|child_id| match child_id {
                Some(id) => id.to_string(),
                None => "N".to_owned(),
            })
            .collect();

        format!("{},{},{},{},{}", 
            self.value.to_string(),
            self.grad.to_string(),
            self.op as u8,
            child_save_ids[0],
            child_save_ids[1],
        ) 
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Op {
    None,
    Add,
    Multiply,
}