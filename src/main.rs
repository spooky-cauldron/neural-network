use neural_network::value_db::ValueDb;


fn main() {
    let mut db = ValueDb::new();

    let a = db.push(1.0);
    let b = db.push(2.0);
    let c = db.op_add(a, b);

    let d = db.push(4.0);
    let e = db.op_mul(c, d);

    db.zero_grad();
    db.backward(e);

    println!("{:?}", db.get(c));
    println!("{:?}", db.get(e));

    db.save("./saves/model_0.txt")
}

