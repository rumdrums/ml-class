function centroids = kMeansRandomCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% I don't yet get why, but any larger than 7, this error returned:

% K-Means iteration 1/20...
% error: computeCentroids: A(I,J,...) = X: dimensions mismatch
% error: called from
%     computeCentroids at line 31 column 20
%     runkMeans at line 54 column 15
%     ex7 at line 97 column 16

scaleVal = 7;
centroids = rand(K, size(X, 2)) * scaleVal;


end

