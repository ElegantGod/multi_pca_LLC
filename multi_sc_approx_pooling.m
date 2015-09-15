function beta = multi_sc_approx_pooling(feaSet, Bset, Cset, pyramid, knn)
%==========================================================================
% Written by chengwei
% July, 2015
%==========================================================================
sub_num = length(Bset);
sub_dim = size(Bset{1},2);
nSmp = size(feaSet.feaArr, 2);
img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);
coding = cell(sub_num, nSmp);
IDX = zeros(nSmp, knn);
dSize = sub_num*sub_dim;
sc_codes = zeros(sub_num*sub_dim, nSmp);
for ii = 1:nSmp
    reg_error = zeros(sub_num,1);
    y = feaSet.feaArr(:,ii);
    for jj = 1:sub_num
        B = Bset{jj};
        center = Cset(:,jj);
        coding{jj,ii} = B'*(y-center);
        reg_error(jj) = norm(y-center);
        %reg_error(jj) = norm(y-center-Bset{jj}*coding{jj,ii});
    end
    [~, idx] = sort(reg_error, 'ascend');
    IDX(ii,:) = idx(1:knn);
end

for ii = 1:nSmp
    idx = IDX(ii, :);
    for jj = 1:knn
        sc_codes((idx(jj)-1)*sub_dim+1:idx(jj)*sub_dim, ii) = coding{idx(jj), ii};
    end
end
sc_codes = abs(sc_codes);
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      
        beta(:, bId) = max(sc_codes(:, sidxBin), [], 2);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));


end