import os 


def read_factor_graph():

    dirname = os.path.dirname(__file__)
    nam = '../scenarios/Sumo'
    scenario_path = os.path.join(dirname, nam)
    factor_graph = {}
    factor_graph_file = os.path.join(scenario_path,
                                         "factor_graph.txt")
    with open(factor_graph_file, "r") as f:
        for line in f:
            entry = line.split(":")
            factor_graph[int(entry[0])] = [x.strip() for x in entry[1].strip().split(",")]
    return factor_graph


if __name__: '__main__'

x = read_factor_graph()
print(x)

