#!/usr/bin/env python3
import os.path
import random
import csv
import config
import utils

def calc(op, vec):
    match op:
        case 'and':
            return utils.bitAnd(vec)
        case 'or':
            return utils.bitOr(vec)
        case 'xor':
            return utils.bitXor(vec)
        case _:
            raise RuntimeError(f'Invalid op {op}')


def createData(mode):
    data = []
    for i in range(len(config.LogicOps)):
        op = config.LogicOps[i]
        data_size = config.TrainSize[i] if mode == 'train' else config.TestSize[i]
        for j in range(data_size):
            vec = []
            for k in range(config.VecLen):
                vec.append(random.randrange(0, 10000) % 2)
            res = calc(op, vec)
            data.append((op, vec, res))
    random.shuffle(data)
    return data


if __name__ == '__main__':
    train_data = createData('train')
    
    data_path = os.path.join(config.RootDir, "data.csv")
    with open(data_path, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        for item in train_data:
            row = item[1]
            row.append(item[0])
            row.append(item[2])
            writer.writerow(row)
            
    test_data = createData('test')
    
    data_path = os.path.join(config.RootDir, "test_data.csv")
    with open(data_path, 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        for item in test_data:
            row = item[1]
            row.append(item[0])
            row.append(item[2])
            writer.writerow(row)