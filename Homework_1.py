from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

knee = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
optimal_k_elbow = knee.elbow

plt.figure(figsize=(8, 5))
plt.axvline(optimal_k_elbow, color='r', linestyle='--')
plt.plot(k_range, inertia, marker='x')
plt.xlabel('Число кластеров')
plt.ylabel('Сумма квадратов ошибок')
plt.grid(True)
plt.show()

print(f"Оптимальное число кластеров по локтю: {optimal_k_elbow}")

best_silhouette = 0
best_score = -1

if __name__ == "__main__":
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        # print(f"k={k}, Silhouette Score = {score:.4f}") ## тут просто все score
        if score > best_score:
            best_score = score
            best_silhouette = k

    print(f"Оптимальное число кластеров по Silhouette Score: {best_silhouette}")


###########################################

import random
import math
import itertools
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

random.seed(42)


def compute_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def initialize_centroids(X, k):
    return random.sample(X, k)


def assign_clusters(X, centroids):
    labels = []
    for point in X:
        distances = [compute_distance(point, centroid) for centroid in centroids]
        min_index = distances.index(min(distances))
        labels.append(min_index)
    return labels


def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = [X[j] for j in range(len(X)) if labels[j] == i]
        if cluster_points:
            mean = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
        else:
            mean = [0] * len(X[0])
        new_centroids.append(mean)
    return new_centroids


def compute_scores(X, labels, centroids):
    score = 0
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        score += sum((a - b) ** 2 for a, b in zip(X[i], centroid))
    return score


def kmeans(X, k, max_iters=100, tol=1e-4):
    global labels
    centroids = initialize_centroids(X, k)
    images = []
    output_dir = "kmeans_output_images" 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i in range(k):
            cluster_points = [X[j] for j in range(len(X)) if labels[j] == i]
            ax.scatter([p[0] for p in cluster_points], [p[1] for p in cluster_points], label=f'Cluster {i}', color=colors[i], s=20)
            
        ax.scatter([centroid[0] for centroid in centroids], [centroid[1] for centroid in centroids], marker='x', color='black', label='Centroids', s=100)
        ax.set_title(f'Iteration {iteration + 1}')
        ax.legend()
        ax.grid(True)

        image_filename = os.path.join(output_dir, f"iteration_{iteration}.png")
        plt.tight_layout()
        plt.savefig(image_filename)
        plt.close(fig)

        images.append(image_filename)

        diff = sum(compute_distance(a, b) for a, b in zip(centroids, new_centroids))
        if diff < tol:
            break
        centroids = new_centroids

    with imageio.get_writer('kmeans_steps.gif', mode='I', duration=0.5) as writer:
        for image in images:
            img = imageio.imread(image)
            writer.append_data(img)

    return centroids, labels


def plot_cluster_projections(X, labels):
    feature_names = ["длина чашелистика", "ширина чашелистика", "длина лепестка", "ширина лепестка"]
    pairs = list(itertools.combinations(range(4), 2))
    colors = ['red', 'blue', 'green']

    plt.figure(figsize=(15, 10))
    for idx, (i, j) in enumerate(pairs):
        plt.subplot(2, 3, idx + 1)
        for label in set(labels):
            xs = [X[p][i] for p in range(len(X)) if labels[p] == label]
            ys = [X[p][j] for p in range(len(X)) if labels[p] == label]
            plt.scatter(xs, ys, label=f'Cluster {label}', color=colors[label], s=20)
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.grid(True)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data.tolist()
    final_k = 3  # взял 3, как в 1 варианте
    centroids, labels = kmeans(X, final_k)
    plot_cluster_projections(X, labels)
