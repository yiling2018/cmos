function [ K ] = normaliseKernel( K, K_te, K_tr )
% original kernel matrix stored in variable K
% output uses the same variable K
% D is a diagonal matrix storing the inverse of the norms
if (~exist('K_tr','var'))
    D = diag(1./sqrt(diag(K)));
    K = D * K * D;
else
    D_te = diag(1./sqrt(diag(K_te)));
    D_tr = diag(1./sqrt(diag(K_tr)));
    K = D_te * K * D_tr;
end

end

