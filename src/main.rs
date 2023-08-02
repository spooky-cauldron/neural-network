use neural_network::value_db::ValueDb;


fn main() {
    let mut db = ValueDb::new();

    let a = db.push(2.0);
    let b = db.push(-3.0);
    let c = db.push(10.0);
    let e = db.op_mul(a, b);
    let d = db.op_add(e, c);
    let f = db.push(-2.0);
    let g = db.op_mul(d, f);

    db.zero_grad();
    db.backward(g);

    println!("{:?}", db.get(c));
    println!("{:?}", db.get(e));

    db.save("./saves/model_0.txt")
}

