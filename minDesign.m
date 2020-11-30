%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Minimizer function

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script finds optimum design point from GP space

% Open issues: 	(1) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [designSpace] = minDesign(probDesired, steps, x, y, hyp, meanfunc, covfunc, inffunc, likfunc)
    [~,f]       = size(x);
    
    minX        = round(min(x),1);
    maxX        = round(max(x),1);
    midX        = round(median(x),2);
    stepX       = (maxX-minX)/steps;
    
    gridVec     = zeros(f, steps+1);
    
    for j = 1:f
        gridVec(j,:) = minX(j):stepX(j):maxX(j);
    end
    
    t   = transpose(...
        combvec(gridVec(1,:), gridVec(2,:), gridVec(3,:), gridVec(4,:)));
    n   = length(t);
    
    [~,~,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
        x, y, t, ones(n, 1));
    
    minSpace    = [t exp(lp)];
    
    designSpace = minSpace((minSpace(:, end) <= probDesired),:);
end