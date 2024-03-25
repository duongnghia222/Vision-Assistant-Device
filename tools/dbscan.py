import numpy as np


def dbscan_straight_cluster(data, min_size):
    labels = np.zeros(data.shape, dtype=int)
    cluster_id = 1

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 0:
                continue  # Skip empty cells
            if labels[i, j] != 0:
                continue  # Point already visited
            neighbors = immediate_neighbors(data, (i, j))
            if len(neighbors) == 0:
                continue  # Point has no neighbors
            if len(neighbors) + 1 >= min_size:
                expand_cluster(data, labels, (i, j), neighbors, cluster_id, min_size)
                cluster_id += 1

    return labels


def immediate_neighbors(data, point):
    neighbors = []
    x, y = point
    for i in range(max(0, x - 1), min(data.shape[0], x + 2)):
        for j in range(max(0, y - 1), min(data.shape[1], y + 2)):
            if data[i, j] != 0 and (i, j) != point:
                neighbors.append((i, j))
    return neighbors


def expand_cluster(data, labels, point, neighbors, cluster_id, min_size):
    labels[point[0], point[1]] = cluster_id
    while neighbors:
        current_point = neighbors.pop(0)
        if labels[current_point[0], current_point[1]] == 0:
            current_neighbors = immediate_neighbors(data, current_point)
            if len(current_neighbors) + 1 >= min_size:
                labels[current_point[0], current_point[1]] = cluster_id
                neighbors.extend(current_neighbors)


# Example usage
data = np.array([[0, 0, 0, 0, 0, 0],
                 [0, 98, 82, 87, 2, 4],
                 [0, 99, 98, 2, 87, 0],
                 [0, 99, 98, 78, 87, 0]])

min_size = 6  # Minimum cluster size
clustered_array = dbscan_straight_cluster(data, min_size)
print("Clustered array:")
print(clustered_array)
