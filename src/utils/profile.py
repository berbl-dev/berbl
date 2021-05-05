import pandas as pd
import re


def read_profile(file):
    """
    Read cProfile into a Pandas DataFrame, (unsafely!) skipping any output in
    the file before the profiling table.
    """
    with open(file, "r") as f:
        r = re.compile("function calls")
        l = f.readline()
        while not r.search(l):
            l = f.readline()

        header = [l]
        r = re.compile("ncalls")
        l = f.readline()
        while not r.search(l):
            header.append(l)
            l = f.readline()
        header.append(l)

        body = f.readlines()

    # We hardcode labels for simplicity's sake (for now).
    columns = [
        "ncalls", "tottime", "tottime/ncalls", "cumtime",
        "cumtime/primitive calls", "filename:lineno(function)"
    ]
    exclude = ["tottime/ncalls", "cumtime/primitive calls"]
    n = len(columns)
    data = [_.split() for _ in body]
    data = [
        row if len(row) == n else row[:n - 1] + [" ".join(row[n - 1:])]
        for row in data
    ]
    data = pd.DataFrame(data, columns=columns)
    for e in exclude:
        del data[e]

    data = data.dropna()
    data["tottime"] = data["tottime"].apply(float)
    data["cumtime"] = data["cumtime"].apply(float)
    return data
