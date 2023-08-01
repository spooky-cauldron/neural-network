use neural_network::value_db::ValueDb;
use neural_network::ID;
use std::collections::HashMap;

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
