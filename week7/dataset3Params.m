function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
lowestErrs = 99999999;
vals = [ 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30 ];

for i = 1:size(vals,1);
	for j = 1:size(vals,1);
		%fprintf('\nC = %d, sigma = %d\n',vals(i),vals(j));
		model = svmTrain(X, y, vals(i), @(x1, x2) gaussianKernel(x1, x2, vals(j)));
		predictions = svmPredict(model, Xval);
		errors = mean(double(predictions ~= yval));
		%fprintf('Error = %d\n',errors);
		if (errors < lowestErrs)
			C = vals(i);
			sigma = vals(j);
			lowestErrs = errors;
		endif
	end
end

fprintf('\nChoosing C=%d, sigma=%d\n',C,sigma);

% =========================================================================

end
