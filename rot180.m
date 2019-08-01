function B = rot180(A)
% ½«¾ØÕóĞı×ª180¶È.
[m,n] = size(A);
B = A(m:-1:1,n:-1:1);
end
