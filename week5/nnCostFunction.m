function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

debug = 0;

a1 = [ones(size(X,1), 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;

%new_y = zeros(size(y,1),10);
%% convert y to matrix:
%for i = 1:size(y,1)
%	new_y(i,y(i)) = 1;
%end
%disp("new_y");
%disp(size(new_y));
%y = new_y;
%
% this doesn't cause dimension error when running check function,
% y = new_y above does:
%disp(y)
%y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

% test:
y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

% no transposing y here, just
% .* multiplication:
cost = (1/m) * sum(sum(( -y.*log(h) - (1-y).*log(1-h) )));
% replace 1st columns with zeros:
Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;
regularization = (lambda/(2*m)) * (sum(sum((Theta1_reg.*Theta1_reg))) ... 
	+ sum(sum((Theta2_reg.*Theta2_reg))));
J = cost + regularization;

if ( debug == 1) 
	whos
endif;

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));


% remove bias column entirely:
Theta2_new = Theta2(:,2:end);

% backprop in a for loop:
for t = 1:m
	% forward prop:
	a1t = [1 X(t,:)];
	z2t = a1t * Theta1';
	a2t = sigmoid(z2t);
	a2t = [1 a2t ];
	z3t = a2t * Theta2';
	a3t = sigmoid(z3t);
	% back prop:
	d3t = a3t - y(t,:);
	delta2 = delta2 + d3t'*a2t;
	d2t = (d3t * Theta2_new) .* sigmoidGradient(z2t);
	delta1 = delta1 + d2t'*a1t;
end;

Theta2_grad = delta2/m;
Theta1_grad = delta1/m;

% regularization:
%fprintf(['Theta2_grad -- BEFORE %d \n'], size(Theta2_grad));
%Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ( Theta2_grad(:,2:end) * (lambda/m) );
%Theta2_grad(:,2:end) = Theta2_grad(:,2:end) * (lambda/m);
%fprintf(['Theta2_grad -- AFTER %d \n'], size(Theta2_grad));
%Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ( Theta1_grad(:,2:end) * (lambda/m) );
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2(:,2:end) * (lambda/m);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1(:,2:end) * (lambda/m);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
