function m = softmax(m)
%% softmax 函数：将输入值映射到[0, 1].
em = exp(m);
m = em./sum(em);
end
