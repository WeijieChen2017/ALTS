def str_to_tuple(s: str):
    return tuple(map(int, s.strip("()").split(",")))