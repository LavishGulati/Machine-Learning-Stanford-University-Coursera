function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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
hTheta = sigmoid(X*theta);
A = (-1).*(y).*(log(hTheta));
A = A + (-1).*(1.-y).*(log(1.-hTheta));
J = sum(A)/m;

B = theta;
B(1) = [0];
B = B.^2;
B = (lambda*sum(B))/(2*m);
J = J + B;

C = (sum((hTheta-y).*X))';
grad = C./m;
D = theta;
D(1) = [0];
D = (lambda.*D)/m;
grad = grad + D;
% =============================================================

end
