%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Minimizer function

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script finds optimum design point from GP space

% Open issues: 	(1) gp could be parallelized for fine grids

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [designSpace, designPoint, minidx] = minDesign(probDesired, steps, x, y, w, ...
    hyp, meanfunc, covfunc, inffunc, likfunc)
    [~,f]       = size(x);
    
    minX        = round(min(x),2);
    maxX        = round(max(x),2);
%     minX        = round(prctile(x,5), 2);
%     maxX        = round(prctile(x,95), 2);
    midX        = round(median(x),2);
    stepX       = (maxX-minX)/steps;

    gridVec     = cell(1, f);
    
    for j = 1:f
        gridVec{j} = minX(j):stepX(j):maxX(j);
    end
    
    % higher resolution for gapRatio
    gridVec{1} = linspace(0.02,0.04,steps+1);
    
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

    % if ties in cost, find the lowest failure design
    minCost         = min(penVec);
    cheapDesigns    = designSpace(penVec == minCost, :);
    [~, minidx]     = min(cheapDesigns(:, end));
    designPoint     = cheapDesigns(minidx, :);
    
    % [~, minidx] = min(penVec);
    % designPoint     = designSpace(minidx, :);
end