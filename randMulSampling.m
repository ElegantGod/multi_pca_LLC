function X = randMulSampling(database, num)
% sample local features for supervised codebook training
% input:   training: all the features
%          num:      the number of sampled features for each class
% output:  X:        the sampled features containing num cell

nclass = database.nclass;
X = cell(nclass,1);

load(database.path{1});
dFea = size(feaSet.feaArr, 1);

for ii = 1:nclass
    % 根据每个类特征的数目相应的改变num
    clabel = find(database.label == ii);
    num_img = length(clabel);
    num_per_img = round(num/num_img);
    num = num_per_img*num_img;
    data = zeros(dFea, num);
    cnt = 0;
    for jj = 1:num_img
        fpath = database.path{clabel(jj)};
        load(fpath);
        num_fea = size(feaSet.feaArr, 2);
        rndidx = randperm(num_fea);
        data(:, cnt+1:cnt+num_per_img) = feaSet.feaArr(:, rndidx(1:num_per_img));
        cnt = cnt+num_per_img;
    end
    X{ii} = data;
end
end