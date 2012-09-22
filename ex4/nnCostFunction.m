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

% Dimensions
% Theta1: hiddenLayerUnits x (n + 1)
% Theta2: outputUnits x (hiddenLayerUnits + 1)
% X: m x (n + 1)
% y: m x 1

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

for i=1:m,
    x_i = [1; X(i, :)']; % (n + 1) x 1
    y_i = zeros(num_labels, 1); y_i(y(i)) = 1;
    h = sigmoid(Theta2 * [1; sigmoid(Theta1 * x_i)]);
    J += sum(y_i .* log(h) + (1 - y_i) .* log(1 - h));
end
J /= -m;
% regularization
regTerms = sum(sum(Theta1 .^ 2)) + sum(sum(Theta2 .^ 2)) - sum(Theta1(:, 1) .^ 2) - sum(Theta2(:, 1) .^2);
regTerms *= lambda / (2 * m);
J += regTerms;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
Delta2 = zeros(num_labels, hidden_layer_size + 1);
Delta1 = zeros(hidden_layer_size, input_layer_size + 1);
for i=1:m,
    x_i = [1; X(i, :)'];
    y_i = zeros(num_labels, 1); y_i(y(i)) = 1;
    z2 = Theta1 * x_i;
    a3 = sigmoid(Theta2 * [1; sigmoid(z2)]);
    a2 = [1; sigmoid(z2)];
    delta3 = a3 - y_i;
    Delta2 += delta3 * a2'; % aggregation
    % think of g'(z2) = sigmoidGradient(Theta1 * x_i) as stored in
    % `the left chambers' of each unit in layer 2, so this is the
    % natural way of multiplication when back-proping
    % refer to http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf
    delta2 = Theta2' * delta3 .* [1; sigmoidGradient(z2)];
    Delta1 += delta2(2:end) * x_i'; % delta2(2:end) to remove the bias term
end
Theta1_grad = Delta1 / m; % hidden_layer_size x (input_layer_size + 1)
Theta2_grad = Delta2 / m; % num_labels x (hidden_layer_size + 1)

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% matrix addition
Theta1_grad += [zeros(size(Theta1_grad, 1), 1) Theta1(:, 2:end)] .* (lambda / m);
Theta2_grad += [zeros(size(Theta2_grad, 1), 1) Theta2(:, 2:end)] .* (lambda / m);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
