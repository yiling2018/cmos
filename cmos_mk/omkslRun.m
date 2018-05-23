data_path = '../data/wiki/';
% set parameters
params = struct();

% params.pos = 1; % corresponding
params.pos = 2; % the same class

% params.dir = 1; % only x2y
% params.dir = 2; % only y2x
params.dir = 3; % two direction

params.C = 0.08; % the aggressiveness parameter
params.margin = 0.1; % margin
params.tri_num = 5*10^3; % number of triplets for training
params.kernel_num = 6; % number kernels
params.disc = 0.998; % discount parameter

%% load data
load([data_path, 'data_norzm.mat']); % load text features and labels of the dataset, label_tr, label_te, text_tr, txt_te
N_tr = size(label_tr, 1);
N_te = size(label_te, 1);

%% generate triplets
data_train = gen_tr_tri_uc(label_tr, params);

scores = zeros(N_te, N_te, params.kernel_num);
z = zeros(2*params.tri_num, params.kernel_num);

%% kernel for text
tic
options_ = [];
options_.KernelType = 'Gaussian'; 
options_.t = 1;
Ktr_t = constructKernel(txt_tr,[],options_);
Ktetr_t = constructKernel(txt_te,txt_tr,options_); %construct kernel matrix for texts, Ktetr_t(N_te x N_tr), Ktr_t(N_tr x N_tr)
Ktetr_t = centerKernel(Ktetr_t, Ktr_t);
Ktr_t = centerKernel(Ktr_t);
toc

%% kernel 1 for image AuxLogits
k = 1;
% compute kernel matrix
tr = load([data_path, 'feat_tr_icptv4.mat']);
te = load([data_path, 'feat_te_icptv4.mat']);
img_tr = tr.AuxLogits;
img_te = te.AuxLogits;
img_tr = normalize_row(img_tr, 'l2');
img_te = normalize_row(img_te, 'l2');
[img_tr, img_te] = zero_mean(img_tr, img_te);
options_ = [];
options_.KernelType = 'Gaussian';
options_.t = 1;
Ktr_i = constructKernel(img_tr,[],options_);
Ktetr_i = constructKernel(img_te,img_tr,options_);
Ktetr_i = centerKernel(Ktetr_i, Ktr_i);
Ktr_i = centerKernel(Ktr_i);
% train
[alpha, beta, z(:,k)] = omkslTrain( data_train, Ktr_i, Ktr_t, params );
% project
scores(:,:,k) = okslCScore( data_train, Ktetr_i, Ktetr_t, alpha, beta, params );
fprintf('after kernel %d', k);
toc

%% kernel 2 for image Logits
k = 2;
% compute kernel matrix
img_tr = tr.Logits;
img_te = te.Logits;
img_tr = normalize_row(img_tr, 'l2');
img_te = normalize_row(img_te, 'l2');
[img_tr, img_te] = zero_mean(img_tr, img_te);
options_ = [];
options_.KernelType = 'Gaussian';
options_.t = 1;
Ktr_i = constructKernel(img_tr,[],options_);
Ktetr_i = constructKernel(img_te,img_tr,options_);
Ktetr_i = centerKernel(Ktetr_i, Ktr_i);
Ktr_i = centerKernel(Ktr_i);
% train
[alpha, beta, z(:,k)] = omkslTrain( data_train, Ktr_i, Ktr_t, params );
% project
scores(:,:,k) = okslCScore( data_train, Ktetr_i, Ktetr_t, alpha, beta, params );
fprintf('after kernel %d', k);
toc

%% kernel 3 for image Predictions
k = 3;
% compute kernel matrix
img_tr = tr.Predictions;
img_te = te.Predictions;
img_tr = normalize_row(img_tr, 'l2');
img_te = normalize_row(img_te, 'l2');
[img_tr, img_te] = zero_mean(img_tr, img_te);
options_ = [];
options_.KernelType = 'Gaussian';
options_.t = 1;
Ktr_i = constructKernel(img_tr,[],options_);
Ktetr_i = constructKernel(img_te,img_tr,options_);
Ktetr_i = centerKernel(Ktetr_i, Ktr_i);
Ktr_i = centerKernel(Ktr_i);
% train
[alpha, beta, z(:,k)] = omkslTrain( data_train, Ktr_i, Ktr_t, params );
% project
scores(:,:,k) = okslCScore( data_train, Ktetr_i, Ktetr_t, alpha, beta, params );
fprintf('after kernel %d', k);
toc

%% kernel 4 for image Mixed_5e
k = 4;
% compute kernel matrix
tr = load([data_path, 'featm_tr_icptv4.mat']);
te = load([data_path, 'featm_te_icptv4.mat']);
fea = permute(tr.Mixed_5e, [4, 3, 2, 1]);
fea = reshape(fea, [], size(fea, 4));
img_tr = fea';
fea = permute(te.Mixed_5e, [4, 3, 2, 1]);
fea = reshape(fea, [], size(fea, 4));
img_te = fea';
img_tr = normalize_row(img_tr, 'l2');
img_te = normalize_row(img_te, 'l2');
[img_tr, img_te] = zero_mean(img_tr, img_te);
options_ = [];
options_.KernelType = 'Gaussian';
options_.t = 1;
Ktr_i = constructKernel(img_tr,[],options_);
Ktetr_i = constructKernel(img_te,img_tr,options_);
Ktetr_i = centerKernel(Ktetr_i, Ktr_i);
Ktr_i = centerKernel(Ktr_i);
% train
[alpha, beta, z(:,k)] = omkslTrain( data_train, Ktr_i, Ktr_t, params );
% project
scores(:,:,k) = okslCScore( data_train, Ktetr_i, Ktetr_t, alpha, beta, params );
fprintf('after kernel %d', k);
toc

%% kernel 5 for image Mixed_6h
k = 5;
% compute kernel matrix
fea = permute(tr.Mixed_6h, [4, 3, 2, 1]);
fea = reshape(fea, [], size(fea, 4));
img_tr = fea';
fea = permute(te.Mixed_6h, [4, 3, 2, 1]);
fea = reshape(fea, [], size(fea, 4));
img_te = fea';
img_tr = normalize_row(img_tr, 'l2');
img_te = normalize_row(img_te, 'l2');
[img_tr, img_te] = zero_mean(img_tr, img_te);
options_ = [];
options_.KernelType = 'Gaussian';
options_.t = 1;
Ktr_i = constructKernel(img_tr,[],options_);
Ktetr_i = constructKernel(img_te,img_tr,options_);
Ktetr_i = centerKernel(Ktetr_i, Ktr_i);
Ktr_i = centerKernel(Ktr_i);
% train
[alpha, beta, z(:,k)] = omkslTrain( data_train, Ktr_i, Ktr_t, params );
% project
scores(:,:,k) = okslCScore( data_train, Ktetr_i, Ktetr_t, alpha, beta, params );
fprintf('after kernel %d', k);
toc

%% kernel 6 for image Mixed_7d
k = 6;
% compute kernel matrix
fea = permute(tr.Mixed_7d, [4, 3, 2, 1]);
fea = reshape(fea, [], size(fea, 4));
img_tr = fea';
fea = permute(te.Mixed_7d, [4, 3, 2, 1]);
fea = reshape(fea, [], size(fea, 4));
img_te = fea';
img_tr = normalize_row(img_tr, 'l2');
img_te = normalize_row(img_te, 'l2');
[img_tr, img_te] = zero_mean(img_tr, img_te);
options_ = [];
options_.KernelType = 'Gaussian';
options_.t = 1;
Ktr_i = constructKernel(img_tr,[],options_);
Ktetr_i = constructKernel(img_te,img_tr,options_);
Ktetr_i = centerKernel(Ktetr_i, Ktr_i);
Ktr_i = centerKernel(Ktr_i);
% train
[alpha, beta, z(:,k)] = omkslTrain( data_train, Ktr_i, Ktr_t, params );
% project
scores(:,:,k) = okslCScore( data_train, Ktetr_i, Ktetr_t, alpha, beta, params );
fprintf('after kernel %d', k);
toc

%% compute theta
theta = ones(1, params.kernel_num);
T = params.tri_num;
for t = 1 : 2*T
    for i = 1 : params.kernel_num
        theta(i) = theta(i) * (params.disc ^ z( t, i ));
    end
    theta = theta / sum(theta) * params.kernel_num;
end

% compute score
score = zeros(N_te);
for i = 1 : params.kernel_num
    score = score + theta(i) * (scores( :, :, i ));
end

%% calculate map
fout = fopen('record_omksl.txt', 'a');
% test img2txt
fprintf('[%s] img search txt:\n', datestr(now,31));
fprintf(fout, '[%s] img search txt:\n', datestr(now,31));
test_s_map(score, label_te, label_te, fout);
% test txt2img
fprintf('[%s] txt search img:\n', datestr(now,31));
fprintf(fout, '[%s] txt search img:\n', datestr(now,31));
test_s_map(score', label_te, label_te, fout);
fprintf(fout, '[%s] --------------------------------------\n', datestr(now,31));
