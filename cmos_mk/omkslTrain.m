function [ alpha, beta, z ] = omkslTrain( data_train, Kv, Kt, params )
% online kernel similarity learning for a kernel
% data_train: the matrix for training triplets, T x 3, T is the number of triplets, each row is a triplet
% Kv: kernel matrix of training images, N_tr x N_tr
% Kt: kernel matrix of training texts, N_tr x N_tr
% C: the aggressiveness parameter
% alpha: tau for pi_i
% beta; tau for pi_t
% z: whether or not there are mistakes. the parameter of Hedging strategy

C = params.C;
T = size( data_train, 1 );

alpha = zeros( T, 1 );
beta = zeros( T, 1 );
z = zeros(2*T, 1);

for t = 1 : T
    cid_p = data_train( t, 1 );
    cid_pp = data_train( t, 2 );
    cid_pn = data_train( t, 3 );
    
    % v2t
    Sp = 0;
    Sn = 0;
    for l = 1 : t - 1
        id_p = data_train( l, 1 );
        id_pp = data_train( l, 2 );
        id_pn = data_train( l, 3 );
        alphal = alpha( l );
        betal = beta( l );
        Sp = Sp + alphal*Kv(cid_p, id_p)*(Kt(id_pp, cid_pp)-Kt(id_pn, cid_pp)) + ...
            betal*(Kv(cid_p, id_pp)-Kv(cid_p, id_pn))*Kt(id_p, cid_pp);
        Sn = Sn + alphal*Kv(cid_p, id_p)*(Kt(id_pp, cid_pn)-Kt(id_pn, cid_pn)) + ...
            betal*(Kv(cid_p, id_pp)-Kv(cid_p, id_pn))*Kt(id_p, cid_pn);
    end
    loss = max( 0, params.margin - Sp + Sn );
    if loss > 0
        V_norm = Kv(cid_p,cid_p)*(Kt(cid_pp,cid_pp)+Kt(cid_pn,cid_pn)-2*Kt(cid_pp,cid_pn));
        alpha(t) = min(C, loss/V_norm);
    end
    if Sp - Sn <= 0
        z(2*t-1) = 1;
    else
        z(2*t-1) = 0;
    end
    % t2v
    Sp = 0;
    Sn = 0;
    for l = 1 : t - 1
        id_p = data_train( l, 1 );
        id_pp = data_train( l, 2 );
        id_pn = data_train( l, 3 );
        alphal = alpha( l );
        betal = beta( l );
        Sp = Sp + alphal*Kv(cid_pp, id_p)*(Kt(id_pp, cid_p)-Kt(id_pn, cid_p)) + ...
            betal*(Kv(cid_pp, id_pp)-Kv(cid_pp, id_pn))*Kt(id_p, cid_p);
        Sn = Sn + alphal*Kv(cid_pn, id_p)*(Kt(id_pp, cid_p)-Kt(id_pn, cid_p)) + ...
            betal*(Kv(cid_pn, id_pp)-Kv(cid_pn, id_pn))*Kt(id_p, cid_p);
    end
    id_p = data_train( t, 1 );
    id_pp = data_train( t, 2 );
    id_pn = data_train( t, 3 );
    Sp = Sp + alpha(t)*Kv(cid_pp, id_p)*(Kt(id_pp, cid_p)-Kt(id_pn, cid_p));
    Sn = Sn + alpha(t)*Kv(cid_pn, id_p)*(Kt(id_pp, cid_p)-Kt(id_pn, cid_p));
    loss = max( 0, params.margin - Sp + Sn );
    if loss > 0
        V_norm = Kt(cid_p,cid_p)*(Kv(cid_pp,cid_pp)+Kv(cid_pn,cid_pn)-2*Kv(cid_pp,cid_pn));
        beta(t) = min(C, loss/V_norm);
    end
    if Sp - Sn <= 0
        z(2*t) = 1;
    else
        z(2*t) = 0;
    end
    
    if mod(t, 100) == 0
        fprintf('%d ', t);
        if mod(t, 1000) == 0
            toc
        end
    end
end

end
