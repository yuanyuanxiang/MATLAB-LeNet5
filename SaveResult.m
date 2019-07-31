function Loss = SaveResult(CNN, loss, errs, iter, n)
%% 更新CNN网络权重和残差，并且绘制Loss曲线.
% CNN: cell数组，依次存放A1, A2, A3, ...和 Loss.
% Loss:上一次误差和精度,即CNN最后一个元素.
% errs:误差和精度.
% iter:迭代次数.
% n: 保存间隔.
% 保存格式为"CNN_s000001.mat".
% 袁沅祥，2019-7

Loss = loss;
step = iter + size(Loss, 2);
Loss = [Loss, errs(:, 1:iter)];
if 0 == mod(iter, n)
    CNN{end} = Loss;
    save(['CNN_s', num2str(step, '%06d'), '.mat'], 'CNN');
end

%% 绘制图像
figure(1);
subplot(1, 2, 1);
plot(Loss(1, :), '-b');
title('loss');

figure(1);
subplot(1, 2, 2);
plot(Loss(2, :), '-r+');
hold on;
plot(Loss(3, :), '-b*');
title('acc');
legend('train', 'test', 'Location', 'SouthEast');
pause(1);

end
