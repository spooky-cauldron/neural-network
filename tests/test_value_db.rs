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


    let expected_gradients: HashMap<ID, f32> = HashMap::from([
        (a, 1.0),
        (b, 2.0),
        (c, 3.0),
        (d, 4.0),
        (e, 12.0),
    ]);

    for (value_id, expected_gradient) in expected_gradients {
        assert_eq!(db.get(value_id).value, expected_gradient);
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


    let expected_gradients: HashMap<ID, f32> = HashMap::from([
        (a, 2.0),
        (b, -3.0),
        (c, 10.0),
        (d, 4.0),
        (e, -6.0),
        (f, -2.0),
        (g, -8.0),
    ]);

    for (value_id, expected_gradient) in expected_gradients {
        assert_eq!(db.get(value_id).value, expected_gradient);
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

    let expected_gradients: HashMap<ID, f32> = HashMap::from([
        (a, 4.0),
        (b, 4.0),
        (c, 4.0),
        (d, 3.0),
        (e, 1.0),
    ]);

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

    let expected_gradients: HashMap<ID, f32> = HashMap::from([
        (a, 4.0),
        (b, 4.0),
        (c, 4.0),
        (d, 3.0),
        (e, 1.0),
    ]);

    for (value_id, expected_gradient) in expected_gradients {
        assert_eq!(db.get(value_id).grad, expected_gradient);
    }

    db.zero_grad();

    for value_id in [a, b, c, d, e] {
        assert_eq!(db.get(value_id).grad, 0.0);
    }
}
