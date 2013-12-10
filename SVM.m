function [pc, numVectors] = SVM(XTrain, YTrain, XTest, YTest, degree, C, threshold)
% Support Vector Machine
% This function trains a SVM on the training set. The SVM is then used to predict
% the labels of the test set. Also, the count of the alpha that passed the
% condition of threshold <= alpha_i <= C-threshold is returned.
%
% Copyright (C) 2013  GPLv3
% by joker__ <g.chers ::at:: gmail.com>

N = size(XTrain,1);

% Defining the Kernel Function
% This Kernel is the Vapnik's polynomial,
% it can although be changed, using the
% following anonymous function
kern = @(X,d) (1 + X).^d;

% Defining variables for quadprog
Dy = diag(YTrain);
K = kern(XTrain * XTrain', degree);		% Kernel application to the dot prod of all the vectors
H = (Dy * K * Dy) / 2;
f = -ones(N,1);
A = [];
c = [];
Aeq = YTrain'; 
ceq = 0;
cl = zeros(N,1);
cu = C * ones(N,1);

alpha = quadprog(H,f,A,c,Aeq,ceq,cl,cu);
% indexes when alpha is subject to the constraints:
idx = find(alpha > threshold & alpha < C - threshold);
numVectors = length(idx);

d = 0;
for i=1:length(idx)
    d = d + YTrain(i) - ((alpha.* YTrain)' * K(:,i));
end
d = d / length(idx);    % Take the mean of the computed d

YPred = zeros(size(YTest));
for i=1:length(YTest)
    YPred(i) = sign(((alpha.* YTrain)' * kern(XTrain*XTest(i,:)',degree) + d) / 2);
end

pc = sum(YPred == YTest) / length(YTest);
