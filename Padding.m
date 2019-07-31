function P = Padding(I, n)
%% 对矩阵进行补零，在周围补n圈.
P = zeros(size(I) + 2 * n);
P(1+n:end-n, 1+n:end-n) = I;
end
