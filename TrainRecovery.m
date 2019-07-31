function [CNN, state] = TrainRecovery()
%% 恢复之前的结果，接着进行训练；或者加载现有神经网络.
% n:各层神经元个数，其中按顺序第一个元素为输入层神经元的个数,
% 最后一个元素为输出层神经元的个数，其余元素为隐藏层的神经元个数.
% CNN: cell数组，依次存放A1, A2, A3, ...和 Loss.
% state: 若返回值为true则表示CNN已训练完毕.返回网络精度.
% 袁沅祥，2019-7

CNN = LoadNN();

if isempty(CNN)
    % 从头开始训练.
    CNN = cell(1, 6);

    CNN{1} = cell(1, 6);    % C1卷积层
    for i=1:6
        CNN{1}{i} = rand(5, 5) - 0.5;
    end

    CNN{2} = cell(16, 7);   % C3卷积层
    for i=1:16
        for j=1:6
            CNN{2}{i, j} = rand(5, 5) - 0.5;
        end
    end

    CNN{3} = rand(120, 401) - 0.5;% 全连接
    CNN{4} = rand(84, 121) - 0.5;% 全连接
    CNN{5} = rand(10, 85) - 0.5;% 全连接
    CNN{6} = [];            % loss, acc
end

disp('CNN infomation:'); disp(CNN);

%% 检测此神经网络是否已训练完成.
state = 0;
if isempty(CNN{end})
    return
end
EarlyStopping = 10; %CNN早停条件
loss = CNN{end}(3, 1:end-EarlyStopping);
best = max(loss);
count = 0;
for i = max(length(loss)+1, 1):length(CNN{end})
    if 0 <= CNN{end}(3,i) && CNN{end}(3,i) <= best
        count = count + 1;
        if count == EarlyStopping
            state = best;
        end
    else
        break
    end
end

end
