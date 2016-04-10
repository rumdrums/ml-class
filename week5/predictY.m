function predictY(Theta1, ...
				 Theta2, ....
                 input_layer_size, ...
                 hidden_layer_size, ...
                 num_labels, ...
                 X, y)

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(size(X,1), 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;
[val, p] = max(a3, [], 2);

disp("y");
disp(y');
disp("y_hat");
disp(p');

end
