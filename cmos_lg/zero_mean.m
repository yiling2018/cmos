function [data_train, data_test, data_validation] = zero_mean(data_train, data_test, data_validation)
% input is (samples number x sample dimension)

data_mean = mean(data_train, 1);

data_train = bsxfun(@minus, data_train, data_mean);
data_test = bsxfun(@minus, data_test, data_mean);
if nargin > 2
    data_validation = bsxfun(@minus, data_validation, data_mean);
end

end

