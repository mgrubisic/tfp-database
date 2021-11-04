%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Minimizer function

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script finds optimum design point from GP space

% Open issues: 	(1) gp could be parallelized for fine grids

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [designSpace, designPoint, designFailureSD] = minDesign(probDesired, steps, x, y, ...
    costV, interceptV, hyp, meanfunc, covfunc, inffunc, likfunc)
    %% Create grid and perform GP
    [~,f]       = size(x);
    
    minX        = round(min(x),2);
    maxX        = round(max(x),2);
    stepX       = (maxX-minX)/steps;

    gridVec     = cell(1, f);
    
    for j = 1:f
        gridVec{j} = minX(j):stepX(j):maxX(j);
    end
    
    % higher resolution for gapRatio
    gridVec{1} = linspace(0.8,1.3,steps+1);
    
    t   = transpose(combvec(gridVec{:}));
    n   = length(t);
    
    [~,b,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
    
    minSpace    = [t exp(lp)];
    
    % Find subset that is under 5% failure
    designSpace = minSpace((minSpace(:, end) <= probDesired),:);
    
    %% Section for grid cost calculation
    % translate dim'less vars to vars used in cost calculations
    g           = 386.4;
    
    zetaRef     = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
    BmRef       = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0];
    
    % calculate main variables
    S1          = 1.017;
    Ss          = 2.282;
    Tshort      = S1/Ss;
    Tm          = designSpace(:,2)*Tshort;
    T2          = designSpace(:,3).*Tm;
    zeta        = designSpace(:,4);
    SaTm        = S1./Tm;
    Bm          = interp1(zetaRef, BmRef, zeta);
    moatGap     = designSpace(:,1).*(g*(SaTm./Bm).^2);
    
    % base shear calc: 2 layers of frame per direction
    kM          = (1/g)*(2*pi./Tm).^2;
    Ws          = 2227.5;
    W           = 3037.5;
    Dm          = g*1.017.*Tm./(4*pi^2.*Bm);
    Vb          = (Dm.*kM*2227.5)/2;
    Vst         = (Vb.*(Ws/W).^(1 - 2.5*zeta));
    Vs          = Vst./designSpace(:,5);
    
    %% Section for full variable regression
    costVar     = [moatGap Tm T2 zeta Vs];   
%     penVec      = designSpace(:,1:f)*w';
%     costV       = [costV; 0];
    penVec      = costVar*costV + interceptV;
%     penVec      = designSpace*costV + interceptV;
%     penVec      = designSpace*costV' + interceptV;
    
%     pen = @(paramVec) (paramVec*w');
%     penResult = arrayfun(@(paramVec) pen(paramVec) , ...
%         designSpace(:,1:f));

    %%
    % if ties in cost, find the lowest failure design
    minCost         = min(penVec);
    cheapDesigns    = designSpace(penVec == minCost, :);
    [~, minidx]     = min(cheapDesigns(:, end));
    designPoint     = cheapDesigns(minidx, :);
    cheapVariance   = b(penVec == minCost,:);
    designFailureSD = sqrt(cheapVariance(minidx, :))/2;
    % [~, minidx] = min(penVec);
    % designPoint     = designSpace(minidx, :);
end