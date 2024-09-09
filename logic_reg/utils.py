import functools

def bitAnd(vec):
    return functools.reduce(lambda x, y: x & y, vec)

def bitOr(vec):
    return functools.reduce(lambda x, y: x | y, vec)

def bitXor(vec):
    return functools.reduce(lambda x, y: x ^ y, vec)