import numpy as np
import pandas as pd 
import time
from os import walk
import re



def generate_adjacent_matrix(data) -> np.array:
        """
        Function that generate the adjacent matrices from the data
        input → pd.DataFrame
        output → np.array
        """
        n = data["node"].max()
        #print(len(data))
        #print(data.head())
        adj = np.zeros((n, n))
        for node, neigbour in data[["node", "neighbour"]].values:
            adj[node-1][neigbour-1] = 1
        adj = np.maximum( adj, adj.transpose() )
        return adj


def check_sol(adj_matrix, sol_vect) -> bool:
    """
    Function that check if the solution of the problem is valid
    inputs : np.array, np.array
    output : bool
    """
    n = len(sol_vect)
    for i in range(n):
        color = sol_vect[i]
        for j in range(n):
            if adj_matrix[i][j] == 1:
                if sol_vect[j] == color:
                    return False

    return True

#-------------------------------------------------------------------------------------------------------------------------------------------



def update_dsat(i, col, dsat, color, adj) -> np.array:
    """
    Function that update the dsat matrix when adding a new color col in the ith vertice
    input = i → int, col → int
    return np.array
    """
    n = len(color)
    for j in range(n):
        if adj[i][j] == 1 and color[j] != col:
            dsat[j] += 1
    return dsat            


def color_vertices(vertice, color, adj) -> int:
    """
    Function that add a new valid color in the vertice
    input : vertice → int
    return : int, the new color
    """
    l_col = color[adj[vertice] == 1]
    if len(l_col) == 0:
        return 1

    max_col = int(np.max(l_col))

    for i in range(1, max_col+1):
        if i not in l_col:
            return i
    return max_col + 1


def choose_next(color, dsat, degree):
    """
    function that choose the next vertice to compute
    inoput : None
    return : int, the index of the vertice
    """
    temp_dsat = np.where(color == 0, dsat, 0)
    all_max =  np.argwhere(temp_dsat == np.amax(temp_dsat)).flatten()
    next = all_max[0]
    for m in all_max:
        if degree[m] > degree[next]:
            next = m
    return m

def color_graph(adj):

    n = len(adj)
    #print("other n : ", n)
    degree = np.sum(adj, axis=1)
    dsat = np.zeros(n)
    color = np.zeros(n)

    start_time = time.time()
    i = np.argmax(degree)
    color[i] = 1
    dsat = update_dsat(i, 1, dsat, color, adj)
    for i in range(n):
        vert = choose_next(color, dsat, degree)
        #print(vert)
        color[vert] = color_vertices(vert, color, adj)
        dsat = update_dsat(vert, color[vert], dsat, color, adj)
    end_time = time.time()

    print(color)

    print("validité : ", check_sol(adj, color))

    print("number of colors : ", len(np.unique(color)))

    print("time taken : ", end_time - start_time)



f = []
for (dirpath, dirnames, filenames) in walk("./data"):
    f.extend(filenames)

print(filenames)

for filename in f:
    with open("./data/{}".format(filename), "r") as f:
        print(filename)
        lines = f.readlines()
    node = []
    neigbour = []
    for line in lines:
        if bool(re.search(r"\Ae", line)):
            splitted = line.replace("\n", "").split(" ")
            node.append(int(splitted[1]))
            neigbour.append(int(splitted[2]))
    data = pd.DataFrame(data=np.array([node, neigbour]).T, columns=["node", "neighbour"])
    #print(data.head())
    adjacent_matrix = generate_adjacent_matrix(data)
    color_graph(adjacent_matrix)

