import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from os.path import basename
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(Z):
    """Vẽ biểu đồ phân cấp dendrogram"""
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title('Dendrogram of Image Clustering')
    plt.xlabel('Images')
    plt.ylabel('Distance')
    plot_path = 'static/uploads/dendrogram.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def extract_image_features(image_path):
    """Trích xuất đặc trưng từ hình ảnh (trung bình màu sắc)"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))  # Resize để dễ tính toán
    avg_color_per_row = np.mean(image, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return avg_color

def hierarchical_clustering(image_paths, n_clusters):
    features = []
    for image_path in image_paths:
        feature = extract_image_features(image_path)
        if feature is not None:
            features.append(feature)

    if len(features) == 0:
        raise ValueError("Không có đặc trưng hợp lệ từ các hình ảnh.")

    features = np.array(features)
    Z = linkage(features, method='ward')

    labels = fcluster(Z, n_clusters, criterion='maxclust')

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(basename(image_paths[idx]))

    # Vẽ biểu đồ dendrogram
    dendrogram_path = plot_dendrogram(Z)

    return clusters, dendrogram_path


