<h1 align = "center">Data Mining Project Report</h1>
<p align = "center"><i>Devansh Bhardwaj (210150010) and Subhash Patel (210150017)</i></p>

# Introduction

The main aim of this project was to assign labels of 10 clusters (from 0 to 9) to 11952 graph nodes. We were given the folowing data:

1. `adjacency.csv`: Adjacency matrix of the graph, containing the information related to its structure.
2. `attributes.csv`: 103 Attributes for each node of the graph, giving additional information regarding each node.
3. `seed.csv`: Had 3 seed nodes from each cluster, giving the labels for 30 nodes in total.

We tried multiple approaches like Spectral Clustering and Graph Neural Networks. Now, We will the see the details (pipelines and results) of every approach.

# 1. Our Approach : Spectral Decomposition followed by clustering

<p align="center">
  <img src = "images/spec.jpg" align='center' height = 400 width = 1000 style="display: block; margin: 0 auto">
</p>


* For the adjacency data, we performed **Spectral Clustering**. This involved performing eigen decomposition of the Laplacian matrix followed by selection of smallest 10 eigenvectors.

  Spectral clustering is preferred for its ability to handle non-linearly separable data and detect clusters of varying shapes and sizes, making it suitable for complex datasets where traditional methods like K-means may struggle.

* For the attributes data, we first standardize the data to have 0 mean and standard devation of 1, and then perform **PCA** (preserving 0.9 fraction of variance) to reduce the dimensionality of the attributes.

  PCA is employed to reduce the dimensionality of data, noise reduction, and improved model performance by capturing the most relevant features.

* We then concatenate the normalized attributes and the eigenvectors to get our final embeddings for each node.

* Next we apply **K-means** clustering of the first 10952 nodes. We initialize the cluster centroids by taking mean of the seed points of each cluster.

* After getting the labels for the first 10952 nodes, we train a 2 layer fully connected **Neural Network** on those nodes. The loss used is simply a cross entropy loss.

* Finally, we get the predictions for the last 1000 nodes.

  This approach gave us the best results of **0.24280** on the **old adjacency** data and 0.14390 on the new adjacency data. The nodes in the new adjacency data have 6000 edges on an average, making it more difficult to deal with, as compared to the old adjacency data which had 50 edges per node on an average.

  Rest of the approaches were tested on the new adjacency data only, they might perform well if trained on the old one.



# 2. Failed GNN approaches


## Graph Convolutional Network (GCN) Based


* We used the matrix form of GCN (2 layers) to produce node embeddings of the graph data using both attributes and adjacency data. Here, the D and A matrix are the adjacency data, and the inital H0 are the attributes (after applying PCA) of the nodes.

  <!-- <img src = "images/gnn.jpg" align='center' height = 250 width = 600>   -->

  Update equation (Normal and Matrix Form):
  
$$
h_i^{(l+1)} = \sigma \left( \sum_{j=1}^{N(i)} \frac{a_{ij}}{d_{ii}} (h_j^{(l)} w_{ji}^{(l)}) + h_i^{(l)} b_i^{(l)} \right)
$$


$$
H^{(l+1)} = \sigma(D^{-1} A H^{(l)} W^{(l)} + H^{(l)} B^{(l)})
$$




* For training, we used the seed data provided. We created pairs of seed nodes of each cluster, and tried to maximize their dot product over the sum of dot porducts with other negative samples. For **Negative Sampling**, we used 30 samples for each node. 

  The loss used is (here n's are negative samples):

$$
\log \left( \frac{Z_u \cdot Z_v}{\sum_{n} Z_u \cdot Z_n} \right)
$$


* After training the GCN, we extract the final embeddings from model, and perform K-means clustering on the entire data. Again, we initialize the cluster centroids by taking mean embedding of the seed points of each cluster.

  This approach gave a score of **0.11194**, which might be due to the very high number of edges of each node. This leads to the aggregation of messages from all the nodes of the graph as the 2-hop graph of this data is almost fully connected.

  We tried by randomly selecting a small portion of the neighbours for passing the message, but that also didnt improve the score.


## GraphSage Network Based


* We used inbuilt functions from **stellargraph** library of python, to create positive and negative node pairs for training, by doing **Random Walk** through the graph. Here, positive signify that the nodes are reachable from each other, and negative signifies the opposite.

* We then pass the input adjacency and attributes data of the node pairs through a GraphSAGE netork which produces embeddings of both the nodes. 

$$
h_u^{(l+1)} = \sigma \left( W^{(l)} \cdot \text{CONCAT} \left( h_u^{(l)}, \frac{1}{|\mathcal{N}(u)|} \sum_{v \in \mathcal{N}(u)} h_v^{(l)} \right) \right)
$$

* After the GraphSAGE, a classifier outputs whether the embeddings (output of the GraphSAGE network) of the given node pair are positive or negative.
  
* The classifier and the GraphSAGE network are trained jointly by using a binary cross-entropy loss, and then we extract the embeddings from the GraphSAGE output and perform K-means as was done in previous approaches.

  This approach gave a score of 0.10132 which is a very poor score, and the reason is again the highly dense nature of th graph.

# Kaggle Performance
For getting a better score on the kaggle leaderboard, we made some adjustments to our method, resulting in a significant improvement in performance. Below are the key modifications we made:

 ## Utilization of Normalized Laplacian
In our previous approach, we employed the standard Laplacian matrix for spectral clustering. However, we found that substituting it with the **Normalized Laplacian** yielded better results. The normalized Laplacian matrix proved to be better as it is scale invariant and is not effected by the variance in the degrees  of the nodes ( standard deviation deviation of around 2000 ) , leading to more accurate clustering results. 

 ## Not using Neural Network
Also earlier, we used a neural network (NN) for predicting the labels of the last 1000 noeds. However, we experimented with a simplified approach by directly clustering all the points without a neural network. This approach eliminated the additional complexity introduced by the neural network training and inference process.

 ## Result
The new methodology resulted in a notable increase in the Kaggle score from 0.14390 to **0.2791**.

# Conclusion
In this project, we tried to cluster 11952 graph nodes into 10 clusters using Spectral Clustering and Graph Neural Networks (GNNs). Our main approach involved Spectral Decomposition followed by clustering. We also explored GNNs but faced challenges due to the graph's highly dense nature. By using the Normalized Laplacian for spectral clustering and eliminating the neural network for label prediction, we got a final accuracy score of 0.2791.
