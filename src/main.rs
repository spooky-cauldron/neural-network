use neural_network::value_db::ValueDb;


fn main() {
    let mut db = ValueDb::new();

    // let x1 = db.push(2.0);
    // let x2 = db.push(0.0);
    // let w1 = db.push(-3.0);
    // let w2 = db.push(1.0);
    // let b = db.push(6.8813735870195432);

    // let x1w1 = db.op_mul(x1, w1);
    // let x2w2 = db.op_mul(x2, w2);
    // let x1w1x2w2 = db.op_add(x1w1, x2w2);
    // let n = db.op_add(x1w1x2w2, b);
    // let o = db.op_tanh(n);

    // db.zero_grad();
    // db.backward(o);

    // println!("{:?}", db.get(c));
    // println!("{:?}", db.get(e));


    let a = db.push(-2.0);
    let b = db.push(3.0);
    let d = db.op_mul(a, b);
    let e = db.op_add(a, b);
    let f = db.op_mul(d, e);

    db.zero_grad();
    db.backward(f);

    db.save("./saves/model_0.txt")
}

