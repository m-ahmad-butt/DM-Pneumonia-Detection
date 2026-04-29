import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import pickle

print("-" * 80)
print("CLUSTERING ANALYSIS - 5 TECHNIQUES")
print("-" * 80)

# Paths
FEATURES_DIR = Path(__file__).parent.parent / 'extracted_features'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
CLUSTERING_RESULTS_PATH = Path(__file__).parent.parent / 'clustering_results.pkl'

# STEP 1: LOAD DATA
print("\n" + "-" * 80)
print("STEP 1: LOADING CNN FEATURES")
print("-" * 80)
X_train = np.load(FEATURES_DIR / 'X_train_features.npy')
y_train = np.load(FEATURES_DIR / 'y_train.npy')
X_test = np.load(FEATURES_DIR / 'X_test_features.npy')
y_test = np.load(FEATURES_DIR / 'y_test.npy')

# Combine for clustering
X_all = np.vstack([X_train, X_test])
y_all = np.hstack([y_train, y_test])

print(f"Total samples: {len(X_all)}")
print(f"Features per sample: 1,280 (extracted from MobileNetV2 CNN)")
print(f"\nGround Truth Distribution:")
print(f"  NORMAL:    {np.sum(y_all == 0)} samples ({np.sum(y_all == 0)/len(y_all)*100:.1f}%)")
print(f"  PNEUMONIA: {np.sum(y_all == 1)} samples ({np.sum(y_all == 1)/len(y_all)*100:.1f}%)")
print(f"\nNote: We will cluster into k=2 groups and compare with ground truth")

# K-Medoids implementation
def k_medoids(X, k=2, max_iter=100):
    """K-Medoids clustering implementation using Manhattan distance"""

    # randomly picked objects to form a k-cluster(s)
    np.random.seed(42)
    medoid_indices = np.random.choice(len(X), k, replace=False)
    
    # mediods runs max max_iter
    for iteration in range(max_iter):
        distances = cdist(X, X[medoid_indices], 'cityblock')  # cityblock = Manhattan
        # print(distances)
        labels = np.argmin(distances, axis=1)
        # print(labels)
        # print("------------")
        
        # Update medoids
        new_medoid_indices = []
        for i in range(k):

            # which index is for cluster i
            # if i=1 eg [false true true] -> [1 2] of cluster indices
            cluster_mask = labels == i
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                new_medoid_indices.append(medoid_indices[i]) # no update
                continue
            
            # actual data 1280D
            cluster_data = X[cluster_indices]

            #cost of every point wrt cluster
            costs = np.sum(cdist(cluster_data, X[cluster_indices], 'cityblock'), axis=1)
            best_idx = cluster_indices[np.argmin(costs)]
            new_medoid_indices.append(best_idx)
        
        if np.array_equal(medoid_indices, new_medoid_indices):
            break
        medoid_indices = new_medoid_indices
    
    distances = cdist(X, X[medoid_indices], 'cityblock')
    labels = np.argmin(distances, axis=1)
    return labels

# Purity calculation (Extrinsic Method)
def purity_score(y_true, y_pred):
    """
    Purity measures how well clusters match ground truth labels.
    Higher is better (range: 0 to 1)
    """
    contingency_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    return np.sum(np.max(contingency_matrix, axis=0)) / len(y_true)


# STEP 3: APPLY CLUSTERING ALGORITHMS
print("\n" + "-" * 80)
print("STEP 3: RUNNING CLUSTERING ALGORITHMS")
print("-" * 80)

clustering_methods = {}

# 1. K-Means
print("\n[1/5] K-MEANS CLUSTERING")
print("-" * 40)
print("  Algorithm: Partition-based")
print("  Centroid: Mean of cluster points")
print("  Parameters: k=2, n_init=10")
print("  Running...")

# no of clusters, _ , no of times k-mean
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clustering_methods['K-Means'] = kmeans.fit_predict(X_all)
print(f"  Cluster 0: {np.sum(clustering_methods['K-Means'] == 0)} samples")
print(f"  Cluster 1: {np.sum(clustering_methods['K-Means'] == 1)} samples")
print("   Completed")

# 2. K-Medoids
print("\n[2/5] K-MEDOIDS CLUSTERING")
print("-" * 40)
print("  Algorithm: Partition-based")
print("  Medoid: Actual data point (not mean)")
print("  Distance: Manhattan (L1 distance)")
print("  Parameters: k=2, max_iter=100")
print("  Running...")
clustering_methods['K-Medoids'] = k_medoids(X_all, k=2)
print(f"  Cluster 0: {np.sum(clustering_methods['K-Medoids'] == 0)} samples")
print(f"  Cluster 1: {np.sum(clustering_methods['K-Medoids'] == 1)} samples")
print("   Completed")

# 3. Single Linkage
print("\n[3/5] SINGLE LINKAGE (MIN)")
print("-" * 40)
print("  Algorithm: Hierarchical (Agglomerative)")
print("  Distance: MIN distance between clusters")
print("  Parameters: n_clusters=2, linkage='single'")
print("  Running...")
single_link = AgglomerativeClustering(n_clusters=2, linkage='single')
clustering_methods['Single Linkage'] = single_link.fit_predict(X_all)
print(f"  Cluster 0: {np.sum(clustering_methods['Single Linkage'] == 0)} samples")
print(f"  Cluster 1: {np.sum(clustering_methods['Single Linkage'] == 1)} samples")
print("   Completed")

# 4. Complete Linkage
print("\n[4/5] COMPLETE LINKAGE (MAX)")
print("-" * 40)
print("  Algorithm: Hierarchical (Agglomerative)")
print("  Distance: MAX distance between clusters")
print("  Parameters: n_clusters=2, linkage='complete'")
print("  Running...")
complete_link = AgglomerativeClustering(n_clusters=2, linkage='complete')
clustering_methods['Complete Linkage'] = complete_link.fit_predict(X_all)
print(f"  Cluster 0: {np.sum(clustering_methods['Complete Linkage'] == 0)} samples")
print(f"  Cluster 1: {np.sum(clustering_methods['Complete Linkage'] == 1)} samples")
print("   Completed")

# 5. BIRCH
print("\n[5/5] BIRCH CLUSTERING")
print("-" * 40)
print("  Algorithm: Hierarchical (uses CF-Tree)")
print("  Memory: Efficient for large datasets")
print("  Parameters: n_clusters=2, threshold=0.5")
print("  Running...")
birch = Birch(n_clusters=2, threshold=0.5)
clustering_methods['BIRCH'] = birch.fit_predict(X_all)
print(f"  Cluster 0: {np.sum(clustering_methods['BIRCH'] == 0)} samples")
print(f"  Cluster 1: {np.sum(clustering_methods['BIRCH'] == 1)} samples")
print("   Completed")


# STEP 4: EVALUATE CLUSTERING QUALITY
print("\n" + "-" * 80)
print("STEP 4: EVALUATING CLUSTERING QUALITY")
print("-" * 80)
print("\nEvaluation Metrics:")
print("  1. Silhouette Coefficient (Intrinsic Method)")
print("     - Measures cluster separation and compactness")
print("     - Range: -1 to 1 (higher is better)")
print("     - Close to 1: well-separated clusters")
print("  2. Purity (Extrinsic Method)")
print("     - Agreement with ground truth labels")
print("     - Range: 0 to 1 (higher is better)")

results = []
print("\n" + "-" * 80)
for method_name, labels in clustering_methods.items():
    print(f"\n{method_name}:")
    print("  " + "-" * 35)
    
    # Intrinsic Method: Silhouette Coefficient
    silhouette = silhouette_score(X_all, labels)
    print(f"  Silhouette Coefficient: {silhouette:.4f}")
    
    # Extrinsic Method: Purity
    purity = purity_score(y_all, labels)
    print(f"  Purity:                 {purity:.4f}")
    
    # Show cluster composition
    for cluster_id in [0, 1]:
        cluster_mask = labels == cluster_id
        normal_count = np.sum((labels == cluster_id) & (y_all == 0))
        pneumonia_count = np.sum((labels == cluster_id) & (y_all == 1))
        print(f"  Cluster {cluster_id}: {normal_count} NORMAL, {pneumonia_count} PNEUMONIA")
    
    # Overall score (weighted average)
    overall_score = (silhouette * 0.5) + (purity * 0.5)
    print(f"  Overall Score:          {overall_score:.4f}")
    
    results.append({
        'Method': method_name,
        'Silhouette': silhouette,
        'Purity': purity,
        'Overall Score': overall_score
    })

# STEP 6: SAVE RESULTS
with open(CLUSTERING_RESULTS_PATH, 'wb') as f:
    pickle.dump({'methods': clustering_methods, 'results': results}, f)

print("\n" + "-" * 80)
print("CLUSTERING ANALYSIS COMPLETED!")
print("-" * 80)
