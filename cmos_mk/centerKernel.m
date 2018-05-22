function [ K ] = centerKernel( K, K_tr )
% original kernel matrix stored in variable K
% output uses the same variable K
% -- if K_tr does not exist
% K is of dimension n x n
% D is a row vector storing the column averages of K
% E is the average of all the entries of K
% -- if K_tr exist
% K is of dimension n_te x n_tr
% K_tr is of dimension n_tr x n_tr
if (~exist('K_tr','var'))
    n = size(K, 1);
    D = sum(K, 1) / n;
    E = sum(D) / n;
    J = ones(n, 1) * D;
    K = K - J - J' + E;
else
    n_tr = size(K_tr,1);
    D_tr = sum(K_tr, 1) / n_tr;
    E_tr = sum(D_tr) / n_tr;
    n_te = size(K, 1);
    D = sum(K, 2) / n_tr;
    K = K - repmat(D_tr, n_te, 1) - repmat(D, 1, n_tr) + E_tr;
end

end

