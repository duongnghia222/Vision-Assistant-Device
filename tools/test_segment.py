import numpy as np
from skimage import measure

def segment_high_value_clusters(array, threshold, min_cluster_size):
    # Step 1: Thresholding
    segmented_clusters = []
    segmented_cluster = (array > threshold).astype(np.uint8)
    if np.sum(segmented_cluster) > min_cluster_size:
        segmented_clusters.append(segmented_cluster)
    # Step 2: Connected Component Analysis
    


    return segmented_clusters

# Example usage
array = np.array([[1, 1, 3, 4, 8, 10],
                  [1, 98, 82, 87, 2, 4],
                  [1, 99, 98, 2, 87, 4],
                  [1, 99, 98, 78, 87, 4]])

threshold_value = 50  # Example threshold
min_cluster_size = 3  # Example minimum cluster size

segmented_clusters = segment_high_value_clusters(array, threshold_value, min_cluster_size)
print("Segmented clusters:")
print(segmented_clusters)