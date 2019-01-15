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

hTheta = X*theta;
A = hTheta-y;
A = A.^2;
J = (1*sum(A))/(2*m);

B = theta;
B(1) = [0];
B = B.^2;
B = (lambda*sum(B))/(2*m);
J = J+B;

C = (sum((hTheta-y).*X))';
grad = C./m;
D = theta;
D(1) = [0];
D = (lambda.*D)/m;
grad = grad + D;

% =========================================================================

grad = grad(:);

end
