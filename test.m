%% test: 训练卷积神经网络.
% 需先在cifar-10-batches-mat目录解压数据.

clc; close all;

if ~exist('Train', 'var')
    [Train, Label, Test, Tag] = Prepare;
end

%{
n = size(Train, 3);
p = randperm(n);
S = 8;
load('cifar-10-batches-mat/batches.meta.mat');
for i = 1:S
    for j = 1:S
        k = j + (i-1) * S;
        subplot(S, S, k);
        imshow(Train(:,:,p(k)), []);
        [id, J] = max(Label(:,p(k)));
        xlabel(label_names{J});
    end
end

clear n p S i j k;
%}

CNN = TrainCNN(Train, Label, Test, Tag, 1e-6);
%}