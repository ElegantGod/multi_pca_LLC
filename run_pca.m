function [basis, M] = run_pca(data, k)
[num, dim] = size(data);
M = mean(data)';
C = cov(data);
[V,D] = eig(C);
D = diag(D);
len = length(D~=0);
if(len<k)
    basis = zeros(dim, len);
    for ii = 1:len
        basis(:,ii) = V(:,dim-ii+1);
    end
else
    basis = zeros(dim, k);
    for ii = 1:k
        basis(:,ii) = V(:, dim-ii+1);
    end
end
end