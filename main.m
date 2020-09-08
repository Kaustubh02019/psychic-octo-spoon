clear all;
close all;
clc;

data_1 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\training_feature_matrix.xlsx");
data_2 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\training_output.xlsx");
data_3 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\test_feature_matrix.xlsx");
data_4 = readmatrix("E:\Study\Open Electives\Neural Network and Fuzzy Logic\Assignments\Assignment1\test_output.xlsx");

x_0 = ones(245,1);
x_1 = (data_1(:,1) - mean(data_1(:,1)))/std(data_1(:,1));
x_2 = (data_1(:,2) - mean(data_1(:,2)))/std(data_1(:,2));
y = (data_2 - mean(data_2))/std(data_2);

theta = rand(3,1);
x = [x_0 x_1 x_2];
m = size(y);

% alpha = 0.001:0.001:1
% iterations = 50:10:500

iterations = 150;
alpha = 0.05;

[theta_0, theta, J_history] = batch_gradient_descent(x, y, theta, alpha, iterations);

x_t0 = ones(104,1);
x_t1 = (data_3(:,1) - mean(data_3(:,1)))/std(data_3(:,1));
x_t2 = (data_3(:,2) - mean(data_3(:,2)))/std(data_3(:,2));
y_t = data_4;
x_test = [x_t0 x_t1 x_t2];
z = size(y_t);
y_p = theta(1)*x_test(:,1) + theta(2)*x_test(:,2) + theta(3)*x_test(:,3);

ypredicted = y_p*std(data_4) + mean(data_4);

 MSE = 0;
 for i = 1:z(1)
     MSE = MSE + ((ypredicted(i,1)-y_t(i,1))^2)/z(1);
 end
% 
%  MSE_final = MSE/z(1); 

no_iterations = 1:iterations;
plot(no_iterations, J_history);
plot3(theta_0(3,:), theta_0(2,:), J_history);
