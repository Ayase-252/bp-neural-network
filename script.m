% This script illustrates a regression problem solved by BP neural network.
% 
% Target function:
% $ y=sin(x_1)cos(x_2) $
% Both x_1 and x_2 belong to [-pi, pi].
%
% Network configuration:
% 2 input neurons, 200 hidden neurons, 1 output neuron.
%
% Training set
% 250 points are choose from target function.
%
% Author: Qingyu Deng(bitdqy@hotmail.com)

clear;

% Setup parameters
% MAXINUM training times
training_times = 10000;
% Criterion of training termination.
% Training will be terminated if sum of squared error for validating
% decreased to sqe_crit.
sqe_crit = 0.2;

% Number of neurons in hidden layer.
n_hidden = 60;

% Learning rate.
learning_rate = 0.02;

% DO NOT TOUCH CODE BELOW %

% Setup target function
target_function = @(x_1, x_2) sin(x_1)*cos(x_2)+2;

% Compute sample set
sample_input = coord_generator(-pi:0.12:pi, -pi:0.12:pi);
sample_target = arrayfun(target_function, sample_input(1,:)', sample_input(2,:)')';

% Partition sample set
n_sample = size(sample_input, 2);
rand_perm = randperm(n_sample);
n_training = ceil(n_sample * 0.6);

training_input = sample_input(:,rand_perm(1:n_training));
training_target = sample_target(:, rand_perm(1:n_training));
validate_input = sample_input(:, rand_perm(n_training + 1:n_sample));
validate_target = sample_target(:, rand_perm(n_training + 1:n_sample));

% Setup network
bpnn = BPNeuralNetwork(2, n_hidden, 1, .02);

% Error track
sqe = zeros(1, training_times);
sqe_v = zeros(1, training_times);

fprintf('Begin training\n');
fprintf('Size of training set: %d.\n', size(training_input, 2));
fprintf('Size of validating set: %d.\n', size(validate_input, 2));
fprintf('SSE criterion: %f.\n', sqe_crit);
disp('=============');


% Training
for i = 1:training_times
    bpnn = bpnn.train(training_input, training_target);
    sqe(i) = bpnn.validate(training_input, training_target);
    sqe_v(i) = bpnn.validate(validate_input, validate_target);
    
    fprintf('%d iteration:\n', i);
    fprintf('SSE of training set: %f.\n', sqe(i));
    fprintf('SSE of validating set: %f.\n', sqe_v(i));
    if i >= 2
        fprintf('diff of SSE: %f.\n', sqe(i)-sqe(i-1));
        fprintf('diff of SSE_v: %f.\n', sqe_v(i)-sqe_v(i-1));
    end
    disp('=============');
    if sqe_v(i) < sqe_crit
        break;
    end
end

% Prepare data for plot
plot_margin = -pi:0.1:pi;
plot_input = coord_generator(plot_margin, plot_margin);
n_pinput = length(plot_margin);
plot_output = bpnn.compute(plot_input);
plot_target = arrayfun(target_function, plot_input(1,:)', plot_input(2,:)');

% Plot
subplot(2,3,1);
plot(1:i, sqe(1:i));
title('Sum of squared error of training set');
subplot(2,3,2);
plot(1:i, sqe_v(1:i));
title('Sum of squared error of validating set');

plot_x = reshape(plot_input(1,:), [n_pinput, n_pinput]);
plot_y = reshape(plot_input(2,:), [n_pinput, n_pinput]);
plot_error = abs((plot_output - plot_target') ./ plot_target');
plot_error_z = reshape(plot_error, [n_pinput, n_pinput]);
plot_output_z = reshape(plot_output, [n_pinput, n_pinput]);
plot_target_z = reshape(plot_target, [n_pinput, n_pinput]);

subplot(2,3,4);
surf(plot_x, plot_y, plot_error_z);
axis([-pi, pi, -pi, pi, 0, 0.5]);
title('Error distribution');
xlabel('x_1');
ylabel('x_2');
zlabel('error ratio');
caxis([0,0.1]);
colorbar;

subplot(2,3,5);
surf(plot_x, plot_y, plot_output_z);
title('NN output');
xlabel('x_1');
ylabel('x_2');
zlabel('output');

subplot(2,3,6);
surf(plot_x, plot_y, plot_target_z);
title('Target output');
xlabel('x_1');
ylabel('x_2');
zlabel('y');