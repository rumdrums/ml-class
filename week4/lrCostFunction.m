function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

disp("X");
disp(size(X));

disp("theta");
disp(size(theta));

disp("y");
disp(size(y));

h = sigmoid(X*theta);
disp("h");
disp(size(h));

% 2-step process for removing 1st column from theta...
shift_theta = theta(2:size(theta)); % I think this is probably a bad way to do this
disp("shift_theta");
disp(size(shift_theta));
% ... and then adding it back as zeros:
theta_reg = [0;shift_theta];
disp("theta_reg");
disp(size(theta_reg));

% this should probably be using .* instead -- eg, -y.*log(h):
J = (1/m)*(-y'*log(h) - (1-y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;
disp("J");
disp(size(J));

pause;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);
%disp("grad");
%disp(size(grad));

% =============================================================

grad = grad(:);

end
