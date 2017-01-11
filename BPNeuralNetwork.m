% Back Propagation Neural Network
%
% This class implements a neural network using standard propagation as
% learning algorithm.
% 
% Author: Qingyu Deng(bitdqy@hotmail.com)
% 
% All input vectors(sample points) should be grouped as
%
%           Col 1             Col 2           ....
%  Row 1 elem 1 of v_1    elem 1 of v_2
%  Row 2 elem 2 of v_1    elem_2 of v_2
%   .           .                .
%   .           .                .


classdef BPNeuralNetwork
    
    properties
        w_input_hidden
        w_hidden_output
        o_input
        o_hidden
        o_output
        alpha
        n_input
        n_hidden
        n_output
    end
    
    methods
        function obj = BPNeuralNetwork(n_input, n_hidden, n_output, learning_rate)
            obj.w_input_hidden = rand(n_input + 1, n_hidden);
            obj.w_hidden_output = rand(n_hidden + 1, n_output);   
           
            obj.o_input = zeros(n_input + 1, 1); 
            % Fix the shadow node of biases of nodes of hidden layer
            obj.o_input(1) = 1;
            
            obj.o_hidden = zeros(n_hidden + 1, 1);
            % Fix the shadow node of biases of nodes of output layer
            obj.o_hidden(1) = 1;
            
            obj.o_output = zeros(n_output, 1);
            obj.alpha = learning_rate;
            
            obj.n_input = n_input;
            obj.n_hidden = n_hidden;
            obj.n_output = n_output;
        end
             
        function obj = train(obj, v_inputs, v_targets)
            % Train neural network with training set
            % Inputs and targets in one pair should be column vector in 
            % respective column in v_inputs and v_targets. 
            
            if length(v_inputs) ~= length(v_targets)
                error('Training set is invaild because of the inconsistent demension of input and targets');
            end            
            
            for i = 1: length(v_inputs)
                inputs = v_inputs(:,i);
                targets = v_targets(:,i);
                obj = obj.forward_propagate(inputs);
                obj = obj.back_propagete(targets);
            end
        end
        
        function outputs = compute(obj, v_inputs)
            for i = 1:size(v_inputs, 2)
                inputs = v_inputs(:,i);
                obj = obj.forward_propagate(inputs);
                outputs(:,i) = obj.o_output;
            end
        end
        
        function sum_squared_error = validate(obj, v_inputs, v_targets)
            outputs = obj.compute(v_inputs);
            delta = outputs - v_targets;
            sum_squared_error = 1/2 .* delta * delta';
        end
    end
    methods(Access=private)
        function obj = forward_propagate(obj, v_inputs)
            obj.o_input(2:obj.n_input + 1) = v_inputs;
            
            % Forward propagate from input layer to hidden layer
            raw_o_hidden = obj.w_input_hidden' * obj.o_input;
            obj.o_hidden(2:obj.n_hidden + 1) = activate_function(raw_o_hidden);
            
            % Hidden layer to output layer
            raw_o_output = obj.w_hidden_output' * obj.o_hidden;
            obj.o_output = raw_o_output;
        end
        
        function obj = back_propagete(obj, v_targets)
            % Output layer to hidden layer
            % d_output = arrayfun(@(o, t) o-t, obj.o_output, v_targets);
            d_output = obj.o_output - v_targets;
            d_w_hidden_output = -obj.alpha .* obj.o_hidden * d_output';
            
            % Hidden layer to input layer
            delta_times_omega = obj.w_hidden_output(2:obj.n_hidden+1,:) * d_output;
            % delta_co = arrayfun(@(o) o*(1-o), obj.o_hidden(2:obj.n_hidden+1));
            % Split into 2 statements below to improve performance
            sliced_o_hidden = obj.o_hidden(2:obj.n_hidden+1);
            delta_co = sliced_o_hidden .* (1 - sliced_o_hidden);
            
            d_hidden = diag(delta_co) * delta_times_omega;
            d_w_input_hidden = -obj.alpha .* obj.o_input * d_hidden';
            
            % Weight addjustment
            obj.w_hidden_output = obj.w_hidden_output + d_w_hidden_output;
            obj.w_input_hidden = obj.w_input_hidden + d_w_input_hidden;
        end
    end
end

function val = activate_function(raw)
    val = 1./(1+exp(-raw));
end