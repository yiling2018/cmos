function [ Y ] = normalize_row( X, varargin )
% normalize data by row
% after normalizing, every row will have l1 norm or l2 norm equals to 1
% X patches is (samples number x sample dimension)

normeps = 1e-5;
if nargin >= 2 && strcmp(varargin{1}, 'l2') % l2 norm
    epssumsq = sum(X.^2, 2) + normeps;
    scale = sqrt(epssumsq);
else % l1 norm
    scale = zeros(size(X, 1), 1);
    for i = 1 : size(X, 1)
        scale(i, 1) = norm(X(i, :), 1) + normeps;
    end
end

Y = bsxfun(@rdivide, X, scale);

end

