# Cluster-Based Active Learning [MATLAB]

* This software implements cluster-based active learning in MATLAB, via Dasgupta's and Hsu's (DH) Algorithm.
* The **original paper** for the DH algorithm can be found [here](http://icml2008.cs.helsinki.fi/papers/324.pdf); cite to credit the authors.
* This code was written to reproduce results similar to those presented in an engineering application paper -- details can be found [here](https://www.sciencedirect.com/science/article/pii/S0022460X18305479?via%3Dihub).

If you find any bugs/errors in the code, get in contact via the issues tab.

## Active learning
Active learning is motivated by scenarios where providing labels `Y` for the measured (input) data `X` is infeasible/impractical. The key philosophy is that an algorithm can provide a more accurate mapping from observations in `X` to labels in `Y` if it can select the data from which it learns. In other words, limited to a budget of `n` observations, active algorithms systematically build a training set (`x_train`, `y_train`) in an intelligent and adaptive manner.

## The DH active learning algorithm
Dasgupta's and Hsu's (DH) cluster-adaptive heuristic starts with a hierarchical clustering of the unlabelled data `X`, which divides the feature-space into many partitions. An informative training set is built by directing queries from the hidden labels in `Y` to areas of the feature-space that appear mixed (in terms of labels), while clusters that appear homogeneous are queried less. When appropriate, queried labels can be propagated to any remaining unlabelled instances, using the cluster structure and a majority vote; this process typically associated with semi-supervised learning. A standard supervised classifier can then be learnt from the resulting labelled dataset `xl`. For further information on the algorithm, refer to the [original paper](http://icml2008.cs.helsinki.fi/papers/324.pdf) and [application paper](https://www.sciencedirect.com/science/article/pii/S0022460X18305479?via%3Dihub).

## Examples
3-dimensional toy data and two demo scripts are provided in the repository.

![](images/fig1.png?raw=true)

The data groups are intentionally mixed, with some more separated groups, to present a challenging classification problem.

### Demo 1: single test
Import the data and define the test-set.
```
load('demo_data');
[idx,~,test_idx] = dividerand(length(data),2/3,0,1/3);

% available data to build the training-set
X = data(idx, 1:end-1); 
Y = data(idx, end); 

% test-set
x_test = data(test_idx, 1:end-1);
y_test = data(test_idx, end);
```

**Cluster** the unlabelled input data `X`.
```
[u, ch] = h_cluster(X);
```
`h_cluster` uses the stock MATLAB function `linkage` to build a hierarchical clustering of the input data. Outputs are the clustered data `u` (indexed) for all nodes in the hierarchy, and the list of child nodes `ch` associated with each cluster. For further details, see this [paper](https://www.sciencedirect.com/science/article/pii/S0022460X18305479?via%3Dihub). 

*Note*, if the dataset is large, consider limiting the maximum number of clusters for computational efficiency. See Demo 2 for an example.

Next, apply the DH algorithm for guided sampling (around the cluster structure) to **actively** build an informative training set `xl`, limited to a budget of `n` observations. Labels are queried from the vector of training labels `Y`. *Note*, the labels in `Y` are **hidden** from the learner until they are queried. 
```
n = 180;
B = 3; % batch size
t = n/3; % number of runs

% run the DH learner
[xl, z] = DH_AL(u, ch, B, t, Y);
```
Plot the labelled training set `xl`, including direct queries and propagated labels; the queried data `z` are circled. Notice how queries are directed towards mixed areas of the feature space, and how the labelled data provided by the DH learner are similar to the true dataset (shown above).

![](images/fig2.png?raw=true)

Use the `xl` to learn a supervised classifier. In this test,  naive Bayes classification (MAP)  is used. (Any supervised classifier can be applied.) The performance of the classifier is assessed using a distinct test set (`x_test`, `y_test`). 
```
% define the training-set with DH results
train_idx = xl(:, 1);
x_train  = X(train_idx, :);
y_train = xl(:, 2);

% train/predict with naive Bayes classification
y_pred = NB(x_train, y_train, x_test);

% calculate classification accuracy
acc = sum(y_pred == y_test)/length(y_test)
```
The classification accuracy for active learning is compared to the performance for standard **passive** training, using a random sample of the same sample budget `n`.
```
% define the training-set by a random sample
train_idx = randperm(size(X,1), n);
x_train = X(train_idx, :);
y_train = Y(train_idx);

% train/predict with naive Bayes classification
y_pred = NB(x_train, y_train, x_test);

% calculate classification accuracy
acc = sum(y_pred == y_test)/length(y_test);
```
An example of the typical output for Demo 1 is &nbsp; ![](images/fig4.png?raw=true)

### Demo 2: comparison to passive learning
The procedure for Demo 1 is now applied while increasing the label budget `n`. The classification error `e` is shown for both *active* and *passive* methods. The results for the toy data are shown in the Figure below. The classification error for active learning is generally lower than passive learning, particularly for `n` ~ 180. For a detailed discussion on the performance of the DH learner, see this [application paper](https://www.sciencedirect.com/science/article/pii/S0022460X18305479?via%3Dihub) to engineering data from aircraft experiments.

![](images/fig3.png?raw=true)

**Note**, this demo can take a while to run, due to test repeats. In order reduce the computational cost, the maximum number of clusters analysed can be limited using `h_cluster`.
```
[u, ch] = h_cluster(X, 'max_clusters', 250);
```
This significantly reduces the computation time, by limiting the extensive search across nodes/clusters during pruning refinements.
