function S = Upsampling(I)
%% 对I进行反向池化操作，mean-上采样.
S = zeros(2*size(I));
[m, n] = size(S);
I = I/4;
S(1:2:m-1,1:2:n-1) = I;
S(2:2:m,1:2:n-1) = I;
S(1:2:m-1,2:2:n) = I;
S(2:2:m,2:2:n) = I;
end
