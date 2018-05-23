data_path = '../data/wiki/';
%% --------------------------------------------------------------
tr = load([data_path, 'feat_tr_icptv4.mat']);
te = load([data_path, 'feat_te_icptv4.mat']);
fea_tr = tr.AuxLogits;
fea_te = te.AuxLogits;
fea_tr = normalize_row(fea_tr, 'l2');
fea_te = normalize_row(fea_te, 'l2');
[fea_tr, fea_te] = zero_mean(fea_tr, fea_te);
size(fea_te, 2)
options_=[];
options_.PCARatio = 0.9;
[eigvector,eigvalue] = myPCA(fea_tr,options_);
fea_tr = fea_tr*eigvector;
fea_te = fea_te*eigvector;
size(fea_te, 2)
img_tr1 = fea_tr;
img_te1 = fea_te;
%%
fea_tr = tr.Logits;
fea_te = te.Logits;
size(fea_te, 2)
fea_tr = normalize_row(fea_tr, 'l2');
fea_te = normalize_row(fea_te, 'l2');
[fea_tr, fea_te] = zero_mean(fea_tr, fea_te);
options_=[];
options_.PCARatio = 0.9;
[eigvector,eigvalue] = myPCA(fea_tr,options_);
fea_tr = fea_tr*eigvector;
fea_te = fea_te*eigvector;
size(fea_te, 2)
img_tr2 = fea_tr;
img_te2 = fea_te;
%%
fea_tr = tr.Predictions;
fea_te = te.Predictions;
size(fea_te, 2)
fea_tr = normalize_row(fea_tr, 'l2');
fea_te = normalize_row(fea_te, 'l2');
[fea_tr, fea_te] = zero_mean(fea_tr, fea_te);
options_=[];
options_.PCARatio = 0.9;
[eigvector,eigvalue] = myPCA(fea_tr,options_);
fea_tr = fea_tr*eigvector;
fea_te = fea_te*eigvector;
size(fea_te, 2)
img_tr3 = fea_tr;
img_te3 = fea_te;
%% --------------------------------------------------------------
tr = load([data_path, 'featm_tr_icptv4.mat']);
te = load([data_path, 'featm_te_icptv4.mat']);
fea_tr = permute(tr.Mixed_5e, [4, 3, 2, 1]);
fea_tr = reshape(fea_tr, [], size(fea_tr, 4));
fea_tr = fea_tr';
fea_te = permute(te.Mixed_5e, [4, 3, 2, 1]);
fea_te = reshape(fea_te, [], size(fea_te, 4));
fea_te = fea_te';
fea_tr = normalize_row(fea_tr, 'l2');
fea_te = normalize_row(fea_te, 'l2');
[fea_tr, fea_te] = zero_mean(fea_tr, fea_te);
size(fea_te, 2)
options_=[];
options_.PCARatio = 0.9;
[eigvector,eigvalue] = myPCA(fea_tr,options_);
fea_tr = fea_tr*eigvector;
fea_te = fea_te*eigvector;
size(fea_te, 2)
img_tr4 = fea_tr;
img_te4 = fea_te;
%%
fea_tr = permute(tr.Mixed_6h, [4, 3, 2, 1]);
fea_tr = reshape(fea_tr, [], size(fea_tr, 4));
fea_tr = fea_tr';
fea_te = permute(te.Mixed_6h, [4, 3, 2, 1]);
fea_te = reshape(fea_te, [], size(fea_te, 4));
fea_te = fea_te';
fea_tr = normalize_row(fea_tr, 'l2');
fea_te = normalize_row(fea_te, 'l2');
[fea_tr, fea_te] = zero_mean(fea_tr, fea_te);
size(fea_te, 2)
options_=[];
options_.PCARatio = 0.9;
[eigvector,eigvalue] = myPCA(fea_tr,options_);
fea_tr = fea_tr*eigvector;
fea_te = fea_te*eigvector;
size(fea_te, 2)
img_tr5 = fea_tr;
img_te5 = fea_te;
%%
fea_tr = permute(tr.Mixed_7d, [4, 3, 2, 1]);
fea_tr = reshape(fea_tr, [], size(fea_tr, 4));
fea_tr = fea_tr';
fea_te = permute(te.Mixed_7d, [4, 3, 2, 1]);
fea_te = reshape(fea_te, [], size(fea_te, 4));
fea_te = fea_te';
fea_tr = normalize_row(fea_tr, 'l2');
fea_te = normalize_row(fea_te, 'l2');
[fea_tr, fea_te] = zero_mean(fea_tr, fea_te);
size(fea_te, 2)
options_=[];
options_.PCARatio = 0.9;
[eigvector,eigvalue] = myPCA(fea_tr,options_);
fea_tr = fea_tr*eigvector;
fea_te = fea_te*eigvector;
size(fea_te, 2)
img_tr6 = fea_tr;
img_te6 = fea_te;
%
save([data_path, 'icptv4_multi_norzm_py.mat'], 'img_tr1', 'img_te1', ...
    'img_tr2', 'img_te2', 'img_tr3', 'img_te3', 'img_tr4', 'img_te4', ...
    'img_tr5', 'img_te5', 'img_tr6', 'img_te6');

%% text
load([data_path, 'data_norzm.mat']);
size(txt_te, 2)
options_=[];
options_.PCARatio = 0.9;
[eigvector,eigvalue] = myPCA(txt_tr,options_);
txt_tr = txt_tr*eigvector;
txt_te = txt_te*eigvector;
size(txt_te, 2)
%
save([data_path, 'txt_norzm_py.mat'], 'txt_tr', 'txt_te');