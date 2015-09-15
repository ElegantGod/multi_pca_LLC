% =========================================================================
%
%
% Written by chengwei
% July, 2015
% =========================================================================

clear all; close all; clc;

% -------------------------------------------------------------------------
% parameter setting
pyramid = [1 2 4];     % spatial block structure for the SPM
knn = 5;               % number of neighbor subspaces
c = 10;                % regularization parameter for linear SVM
nRounds = 5;           % in Liblinear package
tr_num = 30;           % number of random test on the dataset
mem_block = 3000;      % maxnum number of testing features loaded each time
smp_num =10000;        % number of sampled feature each class
sub_dim = 20;          % dimension of each subspace
beta = 1e-5;
gamma = 0.15;
num_iters = 50;
% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');
addpath('sparse_coding');
img_dir = 'image/Caltech101';
data_dir = 'data/Caltech101';
fea_dir = 'features_20/Caltech101';

% -------------------------------------------------------------------------
% retrieve the directory of the  database and load the codebook
database = retr_database_dir(data_dir);

Xpath = ['dictionary/randomX'];
Bpath = ['dictionary/multiDic_20'];

% randomly sampling
% X = randMulSampling(database, smp_num);
% save(Xpath, 'X');
try 
    load(Bpath)
catch
    load(Xpath);
    B = cell(length(X),1);
    Center = zeros(128, length(X));
    for i = 1:length(X)
        [B{i}, Center(:,i)] = run_pca(X{i}', sub_dim);
%         B{i} = reg_sparse_coding(X{i}, sub_dim, eye(sub_dim), beta, gamma, num_iters);
    end
    save(Bpath, 'B','Center');
end
%--------------------------------------------------------------------------
% calculate the (sparse) coding feature
dPerFea = sum(sub_dim*pyramid.^2);
dFea = dPerFea*length(B);
nFea = length(database.path);

fdatabase = struct;
fdatabase.path = cell(nFea, 1);        % path for each imge feature
fdatabase.label = zeros(nFea, 1);      % class label for each image feature

try
    load('fdatabase.mat');
catch
    disp('fdatabase.mat not found!');
    for iter1 = 1:nFea
        if ~mod(iter1, 5)
            fprintf('.');
        end
        if ~mod(iter1, 100)
            fprintf('%d images processed\n', iter1);
        end
        fpath = database.path{iter1};
        flabel = database.label(iter1);
        
        load(fpath);
        [rtpath, fname] = fileparts(fpath);
        feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);
        if knn
            fea = multi_sc_approx_pooling(feaSet, B, Center, pyramid, knn);
        else
            fea = multi_sc_pooling(feaSet, B, pyramid, gamma);
        end
        label = database.label(iter1);
        if ~isdir(fullfile(fea_dir, num2str(flabel)));
            mkdir(fullfile(fea_dir, num2str(flabel)));
        end
        save(feaPath, 'fea', 'label');
        
        fdatabase.label(iter1) = flabel;
        fdatabase.path{iter1} = feaPath;       
    end
    save 'fdatabase.mat' fdatabase
end
%--------------------------------------------------------------------------
fprintf('\n Testing...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);
accuracy1 = zeros(nRounds, 1);
accuracy2 = zeros(nRounds, 1);

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end;
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number: %d\n', length(ts_idx));
    
    % load the traing features
    tr_fea = zeros(length(tr_idx), dFea);
    tr_label = zeros(length(tr_idx), 1);
%     ts_fea = zeros(length(ts_idx), dFea);
%     ts_label = zeros(length(ts_idx), 1);
    
    ts_num = length(ts_idx);
    
    for jj = 1:length(tr_idx)
        fpath = fdatabase.path{tr_idx(jj)};
        load(fpath, 'fea', 'label');
        tr_fea(jj, :) = fea';
        tr_label(jj) = label;
    end
    
%     for jj = 1:length(ts_idx)
%         fpath = fdatabase.path{ts_idx(jj)};
%         load(fpath, 'fea', 'label');
%         ts_fea(jj, :) = fea';
%         ts_label(jj) = label;
%     end

% LLC中的分类方法
    tic;
    c = 10;
    options = ['-c ' num2str(c)];
    model = train(double(tr_label), sparse(tr_fea), options);
    
    acc2 = zeros(nclass, 1);
    
    if ts_num < mem_block,
        % load the testing features directly into memory for testing
        [C] = predict(ts_label, sparse(ts_fea), model);
    else
        % load the testing features block by block
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
        curr_ts_label = zeros(mem_block, 1);
        
        C = [];
        ts_label = [];
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            curr_idx = ts_idx(block_idx); 
            
            % load the current block of features
            for kk = 1:mem_block,
                fpath = fdatabase.path{curr_idx(kk)};
                load(fpath, 'fea', 'label');
                curr_ts_fea(kk, :) = fea';
                curr_ts_label(kk) = label;
            end    
            
            % test the current block features
            ts_label = [ts_label; curr_ts_label];
            [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model);
            C = [C; curr_C];
        end
        
        curr_ts_fea = zeros(rem_fea, dFea);
        curr_ts_label = zeros(rem_fea, 1);
        curr_idx = ts_idx(num_block*mem_block + (1:rem_fea));
        
        for kk = 1:rem_fea,
           fpath = fdatabase.path{curr_idx(kk)};
           load(fpath, 'fea', 'label');
           curr_ts_fea(kk, :) = fea';
           curr_ts_label(kk) = label;
        end  
        
        ts_label = [ts_label; curr_ts_label];
        [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model); 
        C = [C; curr_C];        
    end
    for jj = 1:nclass
        c = clabel(jj);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);
        acc2(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end
    accuracy2(ii) = mean(acc2);
    toc;
    disp(['时间' num2str(toc/60) '分']);
end;

fprintf('Mean accuracy1: %f\n', mean(accuracy1));
fprintf('Standard deviation1: %f\n', std(accuracy1));
fprintf('Mean accuracy2: %f\n', mean(accuracy2));
fprintf('Standard deviation2: %f\n', std(accuracy2));
