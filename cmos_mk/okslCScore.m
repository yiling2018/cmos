function [ score ] = okslCScore( data_train, Kte_v, Kte_t, alpha, beta, params )
% calculate the score matrix for a kernel
% data_train: the matrix for training triplets, T x 3, T is the number of triplets, each row is a triplet
% Kte_v: kernel matrix of testing images with training images, N_te x N_tr
% Kte_t: kernel matrix of testing images with training texts, N_te x N_tr
% alpha: tau for pi_i
% beta; tau for pi_t

N_te = size(Kte_v, 1);
img_te_proja = zeros(N_te, params.tri_num);
img_te_projb = zeros(N_te, params.tri_num);
for j = 1 : N_te
    for i = 1 : params.tri_num
        cid_p = data_train( i, 1 );
        cid_pp = data_train( i, 2 );
        cid_pn = data_train( i, 3 );
        img_te_proja(j, i) = alpha(i)*Kte_v(j, cid_p);
        img_te_projb(j, i) = Kte_v(j, cid_pp) - Kte_v(j, cid_pn);
    end
end
txt_te_proja = zeros(N_te, params.tri_num);
txt_te_projb = zeros(N_te, params.tri_num);
for j = 1 : N_te
    for i = 1 : params.tri_num
        cid_p = data_train( i, 1 );
        cid_pp = data_train( i, 2 );
        cid_pn = data_train( i, 3 );
        txt_te_proja(j, i) = Kte_t(j, cid_pp) - Kte_t(j, cid_pn);
        txt_te_projb(j, i) = beta(i)*Kte_t(j, cid_p);
    end
end

score = img_te_proja * txt_te_proja' + img_te_projb * txt_te_projb';

end

