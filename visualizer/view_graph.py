from dataclasses import dataclass
from enum import Enum
from typing import List
from graphviz import Digraph


class Op(Enum):
    NONE = '0'
    ADD = '1'
    MUL = '2'


@dataclass
class Value:
    value: float
    grad: float
    op: Op
    child_0_id: str
    child_1_id: str


def load_model(path: str) -> List[Value]:
    with open(path, 'r') as f:
        data = f.read()
    data = data.split(',')

    n_value_fields = 4
    values = []
    for i in range(0, len(data), n_value_fields):
        value_data = data[i:i + n_value_fields]
        values.append(Value(*value_data))
    print('Model loaded.')
    return values


def create_graph(model: List[Value]) -> Digraph:
    dot = Digraph('model')

    for i, value in enumerate(model):
        dot.node(str(i), label=f'{value.value} - {value.grad}')
        op_node = None
        if value.op == Op.ADD.value:
            op_node = f'+{i}'
            dot.node(op_node, label='+')
        if value.op == Op.MUL.value:
            op_node = f'*{i}'
            dot.node(op_node, label='*')

        if op_node:
            dot.edge(op_node, str(i))

        for child_id in (value.child_0_id, value.child_1_id):
            if child_id != 'N':
                if op_node:
                    dot.edge(child_id, op_node)
                else:
                    dot.edge(child_id, str(i))
    return dot


def main():
    path = '../saves/model_0.txt'
    model = load_model(path=path)
    graph = create_graph(model=model)
    graph.view()


if __name__ == '__main__':
    main()