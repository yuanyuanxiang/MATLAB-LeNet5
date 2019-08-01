function b = hasNaN(m)
%% 判断矩阵m里是否有NaN值.
n = isnan(m);
b = min(n(:));
end
