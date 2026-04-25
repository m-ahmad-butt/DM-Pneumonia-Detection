# Phase 3 Presentation - Pneumonia Detection using CNN and Clustering

---

## SLIDE 1: Title Slide

**Title:** Pneumonia Detection from Chest X-Rays
**Subtitle:** Phase 3 - CNN Model & Clustering Analysis

**Voice Script:**
"Assalam o Alaikum. Welcome to our Phase 3 presentation on Pneumonia Detection from Chest X-Rays using Deep Learning and Clustering techniques. Today we will demonstrate our CNN model and analyze clustering patterns in medical imaging data."

## SLIDE 3: CNN Architecture - MobileNetV2

**Title:** CNN Model Architecture

**Content:**
- **Base Model:** MobileNetV2 (Transfer Learning)
- **Input:** 128×128 RGB images
- **Architecture:**
  - MobileNetV2 (pre-trained on ImageNet)
  - Global Average Pooling
  - Dense Layer (128 neurons, ReLU)
  - Dropout (0.5)
  - Output Layer (1 neuron, Sigmoid)
- **Total Features Extracted:** 1,280 dimensions

**Voice Script:**
"We used MobileNetV2 as our base architecture with transfer learning. The model takes 128×128 RGB images as input. After the MobileNetV2 layers, we added global average pooling, a dense layer with 128 neurons, dropout for regularization, and a sigmoid output for binary classification. The model extracts 1,280 features from each X-ray image."

---

## SLIDE 4: CNN Training Configuration

**Title:** Training Configuration

**Content:**
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, Precision, Recall
- **Callbacks:**
  - Early Stopping (patience=5)
  - Learning Rate Reduction (factor=0.5, patience=3)
- **Epochs:** 20
- **Batch Size:** 32

**Voice Script:**
"For training, we used the Adam optimizer with binary crossentropy loss. We monitored accuracy, precision, and recall. To prevent overfitting, we implemented early stopping with patience of 5 epochs and learning rate reduction. The model was trained for up to 20 epochs with a batch size of 32."

---

## SLIDE 5: CNN Training Results

**Title:** Training Performance

**Content:**
**Final Epoch (Epoch 10):**
- **Training Accuracy:** 95.99%
- **Validation Accuracy:** 94.44%
- **Training Loss:** 0.1518
- **Validation Loss:** 0.2072

**Training Precision:** 99.26%
**Validation Precision:** 98.37%
**Training Recall:** 95.33%
**Validation Recall:** 94.03%

**Observations:**
- No overfitting (validation accuracy close to training)
- High precision (few false positives)
- Good recall (detecting most pneumonia cases)

**Voice Script:**
"Our CNN achieved excellent results. After 10 epochs, training accuracy reached 95.99% and validation accuracy 94.44%, showing no signs of overfitting. The model achieved 99.26% training precision and 98.37% validation precision, meaning very few false positives. Recall of 95.33% on training and 94.03% on validation shows the model successfully detects most pneumonia cases."

---

## SLIDE 6: Learning Curves Analysis

**Title:** Training vs Validation Performance

**Content:**
**Accuracy Trend:**
- Started: 91.32% (training), 93.77% (validation)
- Final: 95.99% (training), 94.44% (validation)
- Steady improvement across epochs

**Loss Trend:**
- Training loss: 0.420 → 0.152 (decreased)
- Validation loss: 0.216 → 0.207 (stable)
- No divergence = No overfitting

**Key Insight:** Model generalizes well to unseen data

**Voice Script:**
"Analyzing the learning curves, we see steady improvement in accuracy from 91% to 96% on training data, while validation accuracy remained stable around 94%. Training loss decreased from 0.42 to 0.15, and validation loss stayed stable around 0.21. The lack of divergence between training and validation curves confirms our model generalizes well without overfitting."
---
# did for descion trees and clustering
## SLIDE 16: Feature Extraction for Clustering

**Title:** CNN Feature Extraction

**Content:**
- **Purpose:** Extract meaningful features for clustering
- **Layer Used:** Global Average Pooling output
- **Feature Vector:** 1,280 dimensions per image
- **Total Samples:** 5,840 X-rays

**Why Feature Extraction?**
- Raw images: 128×128×3 = 49,152 dimensions (too high!)
- CNN features: 1,280 dimensions (manageable)
- Features capture high-level patterns
- Better for clustering algorithms

**Voice Script:**
"Before applying clustering, we extracted features from our trained CNN. Instead of using raw images with 49,152 dimensions, we extracted 1,280-dimensional feature vectors from the global average pooling layer. These features capture high-level patterns learned by the CNN, making them ideal for clustering analysis on all 5,840 X-ray images."

---
## SLIDE 8: CNN + Decision Tree Approach

**Title:** Hybrid Model: CNN + Decision Tree

**Content:**
**Why Decision Tree on CNN Features?**
- CNN extracts 1,280 high-level features
- Decision Tree provides interpretable rules

**Approach:**
1. Train CNN (MobileNetV2)
2. Extract features from Global Average Pooling layer
3. Train Decision Tree on extracted features
4. Compare with Pure CNN
---

## SLIDE 9: Decision Tree Configuration

**Title:** Decision Tree Setup

**Content:**
**Data Split:**
- Training: 5,216 samples (1,341 NORMAL, 3,875 PNEUMONIA)
- Test: 624 samples (234 NORMAL, 390 PNEUMONIA)
- Features: 1,280 dimensions from CNN

**Hyperparameters:**
- max_depth: 10
- min_samples_split: 20
- min_samples_leaf: 10
- **criterion: gini** (measures impurity)

**Splitting Criteria Options:**
- **Gini Impurity:** Gini = 1 - Σ(p_i)² (used in our model)
- **Entropy:** Entropy = -Σ(p_i × log₂(p_i))


**Voice Script:**
"For the Decision Tree, we used 5,216 training samples and 624 test samples, each with 1,280 CNN features. We configured the tree with maximum depth of 10, minimum 20 samples to split, and minimum 10 samples per leaf. For the splitting criterion, we used Gini impurity which measures node impurity. Alternative criteria include Entropy for information gain and Gain Ratio which normalizes information gain."

---
## SLIDE 11: Decision Tree Structure

**Title:** Trained Tree Structure

**Content:**
**Tree Statistics:**
- Total nodes: 169
- Leaf nodes: 85
- Internal nodes: 84
- Tree depth: 10

**Sample Decision Rules (Root Level):**
```
Node 0: [5,216 samples]
├─ If feature_954 ≤ 1.0854:
│  └─ Node 1: [4,147 samples]
│     ├─ If feature_47 ≤ 0.0889: ...
│     └─ Else: ...
└─ Else (feature_954 > 1.0854):
   └─ Node 116: [1,069 samples]
      ├─ If feature_835 ≤ 0.3359: ...
      └─ Else: ...
```

**Voice Script:**
"The trained Decision Tree has 169 total nodes with 85 leaf nodes and a depth of 10. At the root, it first checks feature 954 - if it's less than or equal to 1.0854, it goes to the left branch with 4,147 samples, otherwise to the right with 1,069 samples. Each branch continues splitting based on different CNN features until reaching a leaf node with a final prediction."

## SLIDE 13: Decision Tree Performance

**Title:** Decision Tree Results

**Performance Metrics:**
- **Accuracy:** 73.40%
- **Precision:** 71.88%
- **Recall:** 94.36%
- **F1-Score:** 0.8160
- **PNEUMONIA Recall:** 94.36%

**Key Insight:** High recall (94.36%) - catches most pneumonia cases!

**Voice Script:**
"The Decision Tree achieved 73.40% accuracy on the test set. Looking at the confusion matrix, it correctly identified 90 normal cases and 368 pneumonia cases. The model achieved 71.88% precision and an impressive 94.36% recall. This high recall means the model successfully detects 94% of pneumonia cases, which is crucial in medical diagnosis where missing a positive case could be life-threatening."


## SLIDE 17: Clustering Techniques Overview

**Title:** 5 Clustering Techniques Applied

**Content:**
1. **K-Means** - Partition-based, uses centroids (mean)
2. **K-Medoids** - Partition-based, uses medoids (actual points)
3. **Single Linkage** - Hierarchical, MIN distance
4. **Complete Linkage** - Hierarchical, MAX distance
5. **BIRCH** - Hierarchical, CF-Tree (memory efficient)

**Goal:** Discover natural groupings in X-ray data
**Clusters:** k=2 (to match NORMAL vs PNEUMONIA)

**Voice Script:**
"We applied five different clustering techniques to analyze patterns in our X-ray data. K-Means and K-Medoids are partition-based methods. Single and Complete Linkage are hierarchical methods using minimum and maximum distances respectively. BIRCH uses a CF-Tree structure for memory efficiency. All methods were configured to find 2 clusters to compare with our ground truth labels."

---

## SLIDE 18: K-Means Clustering

**Title:** K-Means Clustering Results

**Content:**
**Algorithm Details:**
- Type: Partition-based
- Centroid: Mean of cluster points
- Distance: Euclidean
- Parameters: k=2, n_init=10

**Results:**
- Cluster 0: 2,656 samples (30 NORMAL, 2,626 PNEUMONIA)
- Cluster 1: 3,184 samples (1,545 NORMAL, 1,639 PNEUMONIA)

**Evaluation:**
- Silhouette Coefficient: 0.0689
- Purity: 0.7303 (73.03%)
- Overall Score: 0.3996

**Voice Script:**
"K-Means clustering partitions data by finding centroids that minimize within-cluster variance. It created two clusters: Cluster 0 with 2,656 samples mostly pneumonia, and Cluster 1 with 3,184 samples more mixed. The silhouette coefficient of 0.0689 indicates moderate cluster separation. Purity of 73% shows reasonable agreement with ground truth labels."

---

## SLIDE 19: K-Medoids Clustering

**Title:** K-Medoids Clustering Results

**Content:**
**Algorithm Details:**
- Type: Partition-based
- Medoid: Actual data point (not mean)
- Distance: Manhattan (L1)
- Parameters: k=2, max_iter=100

**Results:**
- Cluster 0: 2,629 samples (31 NORMAL, 2,598 PNEUMONIA)
- Cluster 1: 3,211 samples (1,544 NORMAL, 1,667 PNEUMONIA)

**Evaluation:**
- Silhouette Coefficient: 0.0629
- Purity: 0.7303 (73.03%)
- Overall Score: 0.3966

**Voice Script:**
"K-Medoids is similar to K-Means but uses actual data points as cluster centers instead of means. We used Manhattan distance which works well in high-dimensional spaces. The results are similar to K-Means with 2,629 and 3,211 samples in each cluster. Silhouette of 0.0629 and purity of 73% show comparable performance to K-Means."

---

## SLIDE 20: Single Linkage Clustering

**Title:** Single Linkage (MIN) Results

**Content:**
**Algorithm Details:**
- Type: Hierarchical (Agglomerative)
- Distance: MIN distance between clusters
- Parameters: n_clusters=2, linkage='single'

**Results:**
- Cluster 0: 5,839 samples (1,575 NORMAL, 4,264 PNEUMONIA)
- Cluster 1: 1 sample (0 NORMAL, 1 PNEUMONIA)

**Evaluation:**
- Silhouette Coefficient: 0.2880 (HIGHEST!)
- Purity: 0.7303 (73.03%)
- Overall Score: 0.5092 (BEST!)

**Observation:** Chain effect - one large cluster, one outlier

**Voice Script:**
"Single Linkage uses minimum distance between clusters for merging. It produced one very large cluster with 5,839 samples and one single outlier. This demonstrates the 'chain effect' common in single linkage. Despite the imbalance, it achieved the highest silhouette coefficient of 0.2880 and the best overall score of 0.5092, indicating the outlier is genuinely different from the main cluster."

---

## SLIDE 21: Complete Linkage Clustering

**Title:** Complete Linkage (MAX) Results

**Content:**
**Algorithm Details:**
- Type: Hierarchical (Agglomerative)
- Distance: MAX distance between clusters
- Parameters: n_clusters=2, linkage='complete'

**Results:**
- Cluster 0: 3,954 samples (1,552 NORMAL, 2,402 PNEUMONIA)
- Cluster 1: 1,886 samples (23 NORMAL, 1,863 PNEUMONIA)

**Evaluation:**
- Silhouette Coefficient: 0.0555
- Purity: 0.7303 (73.03%)
- Overall Score: 0.3929

**Voice Script:**
"Complete Linkage uses maximum distance between clusters, creating more compact groups. It produced more balanced clusters with 3,954 and 1,886 samples. Cluster 1 is predominantly pneumonia with only 23 normal cases. The silhouette of 0.0555 and overall score of 0.3929 show moderate clustering quality."

---

## SLIDE 22: BIRCH Clustering

**Title:** BIRCH Clustering Results

**Content:**
**Algorithm Details:**
- Type: Hierarchical (uses CF-Tree)
- Memory: Efficient for large datasets
- Parameters: n_clusters=2, threshold=0.5

**Results:**
- Cluster 0: 3,833 samples (1,555 NORMAL, 2,278 PNEUMONIA)
- Cluster 1: 2,007 samples (20 NORMAL, 1,987 PNEUMONIA)

**Evaluation:**
- Silhouette Coefficient: 0.0542
- Purity: 0.7303 (73.03%)
- Overall Score: 0.3922

**Advantage:** Memory efficient, single scan of data

**Voice Script:**
"BIRCH uses a CF-Tree structure for memory-efficient clustering, making it ideal for large datasets. It created clusters of 3,833 and 2,007 samples. Cluster 1 is highly pure with 99% pneumonia cases. With a silhouette of 0.0542 and overall score of 0.3922, BIRCH provides good clustering while being computationally efficient."

---

## SLIDE 23: Clustering Evaluation Metrics

**Title:** Evaluation Metrics Explained

**Content:**
**1. Silhouette Coefficient (Intrinsic)**
- Measures cluster separation and compactness
- Range: -1 to 1 (higher is better)
- Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
- a(i) = avg distance within cluster
- b(i) = avg distance to nearest cluster

**2. Purity (Extrinsic)**
- Agreement with ground truth labels
- Range: 0 to 1 (higher is better)
- Purity = (1/N) × Σ max(cluster-class overlap)

**Overall Score:** 50% Silhouette + 50% Purity

**Voice Script:**
"We used two evaluation metrics. Silhouette Coefficient is an intrinsic method measuring how well-separated clusters are, ranging from -1 to 1. It compares average distance within a cluster to the nearest neighboring cluster. Purity is an extrinsic method measuring agreement with ground truth labels. Our overall score combines both metrics equally at 50% each."

---

## SLIDE 24: Clustering Comparison

**Title:** Clustering Methods Comparison

**Content:**
| Method | Silhouette | Purity | Overall Score | Rank |
|--------|-----------|--------|---------------|------|
| Single Linkage | 0.2880 | 0.7303 | **0.5092** | 🥇 1st |
| K-Means | 0.0689 | 0.7303 | 0.3996 | 2nd |
| K-Medoids | 0.0629 | 0.7303 | 0.3966 | 3rd |
| Complete Linkage | 0.0555 | 0.7303 | 0.3929 | 4th |
| BIRCH | 0.0542 | 0.7303 | 0.3922 | 5th |

**Key Findings:**
- All methods achieved same purity (73%)
- Single Linkage has best separation
- Partition methods (K-Means, K-Medoids) similar performance

**Voice Script:**
"Comparing all five methods, Single Linkage achieved the best overall score of 0.5092, primarily due to its high silhouette coefficient. All methods achieved the same purity of 73%, showing consistent agreement with ground truth. K-Means and K-Medoids performed similarly as both are partition-based methods. Complete Linkage and BIRCH had the lowest scores but still provided meaningful clusters."

---