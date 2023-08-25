use neural_network::value_db::ValueDb;
use neural_network::ID;
use std::collections::HashMap;

#[test]
fn test_value_db_value_insert() {
    let mut db = ValueDb::new();

    let a = db.push(1.0);
    let b = db.push(2.0);
    let c = db.op_add(a, b);
    let d = db.push(4.0);
    let e = db.op_mul(c, d);

    let expected_values: HashMap<ID, f32> =
        HashMap::from([(a, 1.0), (b, 2.0), (c, 3.0), (d, 4.0), (e, 12.0)]);

    for (value_id, expected_value) in expected_values {
        assert_eq!(db.get(value_id).value, expected_value);
    }
}

#[test]
fn test_value_db_value_insert_2() {
    let mut db = ValueDb::new();

    let a = db.push(2.0);
    let b = db.push(-3.0);
    let c = db.push(10.0);
    let e = db.op_mul(a, b);
    let d = db.op_add(e, c);
    let f = db.push(-2.0);
    let g = db.op_mul(d, f);

    let expected_values: HashMap<ID, f32> = HashMap::from([
        (a, 2.0),
        (b, -3.0),
        (c, 10.0),
        (d, 4.0),
        (e, -6.0),
        (f, -2.0),
        (g, -8.0),
    ]);

    for (value_id, expected_value) in expected_values {
        assert_eq!(db.get(value_id).value, expected_value);
    }
}

#[test]
fn test_value_db_value_insert_tanh() {
    let mut db = ValueDb::new();

    let x1 = db.push(2.0);
    let x2 = db.push(0.0);
    let w1 = db.push(-3.0);
    let w2 = db.push(1.0);
    let b = db.push(6.8813735870195432);

    let x1w1 = db.op_mul(x1, w1);
    let x2w2 = db.op_mul(x2, w2);
    let x1w1x2w2 = db.op_add(x1w1, x2w2);
    let n = db.op_add(x1w1x2w2, b);
    let o = db.op_tanh(n);

    let expected_values: HashMap<ID, f32> = HashMap::from([
        (x1, 2.0),
        (x2, 0.0),
        (w1, -3.0),
        (w2, 1.0),
        (b, 6.8813735870195432),
        (x1w1, -6.0),
        (x2w2, 0.0),
        (x1w1x2w2, -6.0),
        (n, 0.8813735870195432),
        (o, 0.7071067811865476),
    ]);

    for (value_id, expected_value) in expected_values {
        let delta = (db.get(value_id).value - expected_value).abs();
        assert!(delta < 0.001);
    }
}

#[test]
fn test_value_db_backpropagation() {
    let mut db = ValueDb::new();

    let a = db.push(1.0);
    let b = db.push(2.0);
    let c = db.op_add(a, b);
    let d = db.push(4.0);
    let e = db.op_mul(c, d);

    db.zero_grad();
    db.backward(e);

    let expected_gradients: HashMap<ID, f32> =
        HashMap::from([(a, 4.0), (b, 4.0), (c, 4.0), (d, 3.0), (e, 1.0)]);

    for (value_id, expected_gradient) in expected_gradients {
        assert_eq!(db.get(value_id).grad, expected_gradient);
    }
}

#[test]
fn test_value_db_backpropagation_2() {
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

    let expected_gradients: HashMap<ID, f32> = HashMap::from([
        (a, 6.0),
        (b, -4.0),
        (c, -2.0),
        (d, -2.0),
        (e, -2.0),
        (f, 4.0),
        (g, 1.0),
    ]);

    for (value_id, expected_gradient) in expected_gradients {
        assert_eq!(db.get(value_id).grad, expected_gradient);
    }
}

#[test]
fn test_value_db_backpropagation_tanh() {
    let mut db = ValueDb::new();

    let x1 = db.push(2.0);
    let x2 = db.push(0.0);
    let w1 = db.push(-3.0);
    let w2 = db.push(1.0);
    let b = db.push(6.8813735870195432);

    let x1w1 = db.op_mul(x1, w1);
    let x2w2 = db.op_mul(x2, w2);
    let x1w1x2w2 = db.op_add(x1w1, x2w2);
    let n = db.op_add(x1w1x2w2, b);
    let o = db.op_tanh(n);

    db.zero_grad();
    db.backward(o);

    let expected_gradients: HashMap<ID, f32> = HashMap::from([
        (x1, -1.5),
        (x2, 0.5),
        (w1, 1.0),
        (w2, 0.0),
        (b, 0.5),
        (x1w1, 0.5),
        (x2w2, 0.5),
        (x1w1x2w2, 0.5),
        (n, 0.5),
        (o, 1.0),
    ]);

    for (value_id, expected_gradient) in expected_gradients {
        let delta = (db.get(value_id).grad - expected_gradient).abs();
        assert!(delta < 0.001);
    }
}

#[test]
fn test_value_db_backpropagation_gradient_accumulation() {
    let mut db = ValueDb::new();

    let a = db.push(3.0);
    let b = db.op_add(a, a);

    db.zero_grad();
    db.backward(b);

    let expected_gradients: HashMap<ID, f32> = HashMap::from([(a, 2.0), (b, 1.0)]);

    for (value_id, expected_gradient) in expected_gradients {
        let delta = (db.get(value_id).grad - expected_gradient).abs();
        println!(
            "expected: {}, actual: {}",
            expected_gradient,
            db.get(value_id).grad
        );
        assert!(delta < 0.001);
    }
}

#[test]
fn test_value_db_backpropagation_gradient_accumulation_2() {
    let mut db = ValueDb::new();

    let a = db.push(-2.0);
    let b = db.push(3.0);
    let d = db.op_mul(a, b);
    let e = db.op_add(a, b);
    let f = db.op_mul(d, e);

    db.zero_grad();
    db.backward(f);

    let expected_gradients: HashMap<ID, f32> =
        HashMap::from([(a, -3.0), (b, -8.0), (d, 1.0), (e, -6.0), (f, 1.0)]);

    for (value_id, expected_gradient) in expected_gradients {
        let delta = (db.get(value_id).grad - expected_gradient).abs();
        println!(
            "expected: {}, actual: {}",
            expected_gradient,
            db.get(value_id).grad
        );
        assert!(delta < 0.001);
    }
}

#[test]
fn test_value_db_zero_grad() {
    let mut db = ValueDb::new();

    let a = db.push(1.0);
    let b = db.push(2.0);
    let c = db.op_add(a, b);
    let d = db.push(4.0);
    let e = db.op_mul(c, d);

    db.zero_grad();

    for value_id in [a, b, c, d, e] {
        assert_eq!(db.get(value_id).grad, 0.0);
    }

    db.backward(e);

    let expected_gradients: HashMap<ID, f32> =
        HashMap::from([(a, 4.0), (b, 4.0), (c, 4.0), (d, 3.0), (e, 1.0)]);

    for (value_id, expected_gradient) in expected_gradients {
        assert_eq!(db.get(value_id).grad, expected_gradient);
    }

    db.zero_grad();

    for value_id in [a, b, c, d, e] {
        assert_eq!(db.get(value_id).grad, 0.0);
    }
}
