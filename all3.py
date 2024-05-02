import numpy as np #use for mathematical functions 
import matplotlib.pyplot as plt #use for plotting the graph 
import time #to calculate the execution time 

# Common function to read an adjacency matrix from a file
def read_adjacency_matrix(filename):
    return np.loadtxt(filename, dtype=int) 

# Edmonds-Karp Implementation
def bfs(C, F, source, sink): #c is the graph, F is flow matrix (residual graph) 
    queue = [source]
    visited = [False] * len(C)
    visited[source] = True
    parent = [-1] * len(C)

    for u in queue:
        for v in range(len(C)):
            if not visited[v] and C[u][v] - F[u][v] > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True, parent
    return False, parent

def edmonds_karp(C, source, sink):
    n = len(C) #number of nodes
    F = np.zeros((n, n), dtype=int)
    max_flow = 0

    while True:
        path_found, parent = bfs(C, F, source, sink)
        if not path_found:
            break

        path_flow = float('Inf')
        v = sink
        while v != source:
            path_flow = min(path_flow, C[parent[v]][v] - F[parent[v]][v])
            v = parent[v]

        v = sink
        while v != source:
            u = parent[v]
            F[u][v] += path_flow
            F[v][u] -= path_flow
            v = u

        max_flow += path_flow

    return max_flow

# Ford-Fulkerson Implementation using DFS
def dfs(C, F, source, sink, visited, parent):
    stack = [source]
    visited[source] = True

    while stack:
        u = stack.pop()
        for v in range(len(C)):
            if not visited[v] and C[u][v] - F[u][v] > 0:
                visited[v] = True
                parent[v] = u
                stack.append(v)
                if v == sink:
                    return True
    return False

def ford_fulkerson(C, source, sink):
    n = len(C)
    F = np.zeros((n, n), dtype=int)
    max_flow = 0
    parent = [-1] * n

    while True:
        visited = [False] * n
        if not dfs(C, F, source, sink, visited, parent):
            break

        path_flow = float('Inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, C[u][v] - F[u][v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            F[u][v] += path_flow
            F[v][u] -= path_flow
            v = u

        max_flow += path_flow

    return max_flow

# Greedy Algorithm Implementation
def greedy_algorithm(C, source, sink):
    max_flow = 0
    n = len(C)
    flow = np.zeros((n, n), dtype=int)
    parent = [-1] * n
    parent[source] = -2
    min_capacity = [float('Inf')] * n

    stack = [(source, float('Inf'))]

    while stack:
        u, flow_upto_now = stack.pop()
        for v in range(n):
            if C[u][v] - flow[u][v] > 0 and parent[v] == -1:
                parent[v] = u
                min_capacity[v] = min(flow_upto_now, C[u][v] - flow[u][v])
                if v == sink:
                    increment = min_capacity[sink]
                    max_flow += increment
                    while v != source:
                        u = parent[v]
                        flow[u][v] += increment
                        flow[v][u] -= increment
                        v = u
                    parent = [-1] * n
                    parent[source] = -2
                    min_capacity = [float('Inf')] * n
                    stack = [(source, float('Inf'))]
                    break
                stack.append((v, min_capacity[v]))

    return max_flow

# Function to measure execution times
def measure_execution_times(algorithm, C, num_tests=1):
    times = []
    for _ in range(num_tests):
        source = 0
        sink = len(C) - 1
        start_time = time.time()
        _ = algorithm(C, source, sink)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

# Plotting the execution times
def plot_execution_times(datasets, names, title='Algorithm Comparison'):
    plt.figure(figsize=(12, 6))
    for name, times in datasets.items():
        plt.plot(names, times, marker='o', label=name)
    plt.xlabel('Dataset according to density')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Datasets to test
dataset_files = ['0.1.txt','0.3.txt','0.5.txt','0.7.txt','0.9.txt']
names = ['0.1','0.3', '0.5', '0.7','0.9']
algorithms = {
    'Edmonds-Karp': edmonds_karp,
    'Ford-Fulkerson': ford_fulkerson,
    'Greedy Algorithm': greedy_algorithm
}
execution_times = {alg: [] for alg in algorithms}

for filename in dataset_files:
    C = read_adjacency_matrix(filename)
    for alg_name, alg_func in algorithms.items():
        avg_time = measure_execution_times(alg_func, C, num_tests=1)
        print(filename,avg_time,alg_name)
        execution_times[alg_name].append(avg_time)

plot_execution_times(execution_times, names)
