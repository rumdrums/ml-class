function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

m = size(X,1);
%X = [ones(m, 1) X];
y_hat = X*theta;
sumSquErrs = sum((y_hat-y).^2);
theta = theta(2:size(theta));
theta = [0;theta];
J = sumSquErrs / (2*m) + (lambda/(2*m))*sum(theta.^2) ;
grad =  X' * (y_hat-y)/m + theta*(lambda/m);

% =========================================================================

grad = grad(:);

end
