function CNN = LoadNN(file)
%% 从指定目录加载现有神经网络.
% file: CNN存放目录.
% 返回CNN: cell数组，依次存放A1, A2, A3, ...和 Loss.
% 袁沅祥，2019-7

if nargin == 0
    file = 'CNN_s*.mat';
end

cnn = dir(file);
if ~isempty(cnn)
    load(cnn(end).name);
    fprintf('Load CNN [%s] succeed.\n', cnn(end).name);
else
    CNN = cell(0);
    fprintf('Load CNN Networks failed.\n');
end

end
