%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Minimizer function

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script finds optimum design point from GP space

% Open issues: 	(1) gp could be parallelized for fine grids

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [designSpace, designPoint] = minDesign(probDesired, steps, x, y, w, ...
    hyp, meanfunc, covfunc, inffunc, likfunc)
    [~,f]       = size(x);
    
    minX        = round(min(x),1);
    maxX        = round(max(x),1);
    midX        = round(median(x),2);
    stepX       = (maxX-minX)/steps;

    gridVec     = cell(1, f);
    
    for j = 1:f
        gridVec{j} = minX(j):stepX(j):maxX(j);
    end
    
    t   = transpose(combvec(gridVec{:}));
    n   = length(t);
    
    [~,~,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
        x, y, t, ones(n, 1));
    
    minSpace    = [t exp(lp)];
    
    designSpace = minSpace((minSpace(:, end) <= probDesired),:);
    
%     penVec      = designSpace(:,1:f)*w';
    penVec      = designSpace*w';
    
%     pen = @(paramVec) (paramVec*w');
%     penResult = arrayfun(@(paramVec) pen(paramVec) , ...
%         designSpace(:,1:f));
    
    [~, minidx] = min(penVec);
    
    designPoint     = designSpace(minidx, 1:f);
end