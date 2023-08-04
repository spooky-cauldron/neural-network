use neural_network::value_db::ValueDb;


fn main() {
    let mut db = ValueDb::new();

    let a = db.push(-2.0);
    let b = db.push(3.0);
    let d = db.op_mul(a, b);
    let e = db.op_add(a, b);
    let f = db.op_mul(d, e);

    db.zero_grad();
    db.backward(f);

    db.save("./saves/model_0.txt")
}

