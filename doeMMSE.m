%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Weighted IMSE DOE

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	January 2021

% Description: 	Script finds the IMSE criterion to allow for design of
% experiments to find new points in the isolator data set

% Open issues: 	(1) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

rng(0,'twister');
isolFull    = readtable('../pastRuns/random200withTfb.csv');

% scaling Sa(Tm) for damping, ASCE Ch. 17
g           = 386.4;
zetaRef     = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
BmRef       = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0];
isolFull.Bm  = interp1(zetaRef, BmRef, isolFull.zetaM);

TfbRatio    = isolFull.Tfb./isolFull.Tm;
mu2Ratio    = isolFull.mu2./(isolFull.GMSTm./isolFull.Bm);
mu1Ratio    = isolFull.mu1./(isolFull.GMSTm./isolFull.Bm);
gapRatio    = isolFull.moatGap./(g.*(isolFull.GMSTm./isolFull.Bm).*isolFull.Tm.^2);
T2Ratio     = isolFull.T2./isolFull.Tm;
Ry          = isolFull.RI;
zeta        = isolFull.zetaM;

Tshort      = isolFull.S1/2.282;
TmRatio     = isolFull.Tm./Tshort;

collapsed   = (isolFull.collapseDrift1 | isolFull.collapseDrift2) ...
    | isolFull.collapseDrift3;

collapsed   = double(collapsed);
collapsed(collapsed==0)   = -1;

% x should be all combinations of x1 and x2
% y should be their respective resulting impact boolean
% x           = [mu2Ratio, gapRatio, T2Ratio, zeta, Ry];
% x           = [gapRatio, T2Ratio, mu2Ratio, Ry];
% x           = [mu1Ratio, gapRatio, T2Ratio, zeta, Ry];
% x           = [mu2Ratio, T2Ratio, zeta, Ry];
% x           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];
% x           = [gapRatio, Ry];

% conference paper:
% x           = [gapRatio, T2Ratio, zeta, Ry];

maxDrift    = max([isolFull.driftMax1, isolFull.driftMax2, isolFull.driftMax3], ...
    [], 2);

%% Classification GP

xFull       = [gapRatio, Ry];
% xFull           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];

% limit to 20 points
randSet     = randsample(height(isolFull), 100);
x           = xFull(randSet,:);
xReserve    = xFull(setdiff(1:height(xFull), randSet), :);

y           = collapsed(randSet);
yReserve    = collapsed(setdiff(1:height(xFull), randSet));

k           = length(x);

[~,f]       = size(x);

% try mean as constant
% meanfunc    = @meanConst; hyp.mean = 0;

% try ignoring the mean function
% meanfunc    = [];

% conference paper:
% try mean as affine function
meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [zeros(1,f) 0]';

% try mean as linear function
% meanfunc    = @meanLinear; hyp.mean = [zeros(1,f)]';

% poly?
% meanfunc    = {@meanPoly,2}; hyp.mean = [zeros(1,2*f)]';

covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell*ones(1,f) sf]);
% covfunc     = @covSEiso; hyp.cov = [0 0];

% Logit regression
likfunc     = @likLogistic;
inffunc     = @infLaplace;

% conference paper:
hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
% hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);


%% Regression try GP

xFull       = [gapRatio, Ry];
% xFull           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];

% limit to 20 points
randSet     = randsample(height(isolFull), 100);
x           = xFull(randSet,:);
xReserve    = xFull(setdiff(1:height(xFull), randSet), :);
y           = maxDrift(randSet);
yReserve    = maxDrift(setdiff(1:height(xFull), randSet));

k           = length(x);

[e,f]       = size(x);

% try mean as constant
% meanfunc    = @meanConst; hyp.mean = 0;

% try ignoring the mean function
% meanfunc    = [];

% conference paper:
% try mean as affine function
% meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [zeros(1,f) 0]';

% try mean as linear function
meanfunc    = @meanLinear; hyp.mean = zeros(1,f)';

% poly?
% meanfunc    = {@meanPoly,3}; hyp.mean = [zeros(1,3*f)]';

covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell*ones(1,f) sf]);
% covfunc     = @covSEiso; hyp.cov = [0 0];

% Logit regression
likfunc     = @likGauss; hyp.lik = log(0.1);
inffunc     = @infGaussLik;

% conference paper:
hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
% hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);


%% sequentially add points (option to update hyp)
% Run optimization
lPoints = 20;

% Domain generation
close all;
steps       = 15;
minX        = round(min(x),2);
maxX        = round(max(x),2);

gridVec     = cell(1, f);

for j = 1:f
    gridVec{j} = linspace(minX(j), maxX(j), steps);
end

xs   = transpose(combvec(gridVec{:}));

xsOG    = xs;

[x1, x2]  = meshgrid(gridVec{1}',gridVec{2}');

varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2, k)

% covTracker = zeros(lPoints, f);
% meanTracker = zeros(lPoints, f);
xSuggest    = zeros(lPoints, f);

warning('off')
for i = 1:lPoints
%     xNext   = fminsearchbnd(@fn_imse, firstTry, lb, ub, options, hyp, covfunc, x);
%     xNext   = fminsearchbnd(@fn_imse_gpml, firstTry, lb, ub, options, hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
    [~, xNext]   = fn_mmse_gpml(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    xSuggest(i,:) = xNext;
    
    [xFound, yFound, xReserve, yReserve] = libSearch(xNext, xReserve, yReserve);
    
    x       = [x; xFound];
    y       = [y; yFound];
     
    % redo GP?
    [e,f]       = size(x);
    meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [zeros(1,f) 0]';
%     covfunc     = @covSEiso; hyp.cov = [0 0];
    covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell*ones(1,f) sf]);
    likfunc     = @likGauss; hyp.lik = log(0.1);
    inffunc     = @infGaussLik;
    hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);
    covTracker(i,:) = hyp.cov;
    meanTracker(i,:) = hyp.mean;
    
	% remesh to improve resolution of search
    xs      = remesh(xFull, f, xFound, steps);
end
warning('on')

% % see variance plot
% % redo GP?
% [e,f]       = size(x);
% meanfunc    = @meanLinear; hyp.mean = [zeros(1,f)]';
% covfunc        = @covSEiso; hyp.cov = [0 0];
% likfunc     = @likGauss; hyp.lik = log(0.1);
% inffunc     = @infGaussLik;
% hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);
varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xsOG, x1, x2)
meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xsOG, x1, x2)
evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xsOG, x1, x2, k)

%% domain mesher

function xs     = remesh(x, f, xCenter, steps)
    xRight      = linspace(pi, 3*pi/2, round(steps/2));
    xLeft       = linspace(pi/2, pi, round(steps/2));
    
    minX        = round(min(x),2);
    maxX        = round(max(x),2);
    
    gridVec     = cell(1, f);
    
    for i = 1:f
        leftRange   = xCenter(i) - minX(i);
        rightRange  = maxX(i) - xCenter(i);
            
        leftSpace    = xCenter(i) - (cos(xLeft)+1).*leftRange;
        rightSpace   = xCenter(i) + (cos(xRight)+1).*rightRange;
        
        gridVec{i}   = [leftSpace(1:(end-1)) rightSpace];
    end

    xs   = transpose(combvec(gridVec{:}));
end

%% evolution plot

function evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2, k)
    n = length(xs);
    [~,ys2,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, ...
        xs, ones(n,1));
    
    figure
    contour(x1, x2, reshape(ys2, size(x1))' );
    hold on
    
    xOrig   = x(1:k, :);
    xAdded  = x((k+1):end, :);
    yOrig   = y(1:k, :);
    yAdded  = y((k+1):end, :);
    
    xOrigFail   = xOrig(yOrig == 1, :);
    xOrigOkay   = xOrig(yOrig == -1, :);
    xAddedFail   = xAdded(yAdded == 1, :);
    xAddedOkay   = xAdded(yAdded == -1, :);
    
    scatter(xOrigFail(:,1), xOrigFail(:,2), 'rx')
    scatter(xOrigOkay(:,1), xOrigOkay(:,2), 'ro')
    
    scatter(xAddedFail(:,1), xAddedFail(:,2), 'bx')
    scatter(xAddedOkay(:,1), xAddedOkay(:,2), 'bo')
    
    title('Points added on variance contours')
    xlabel('gap ratio')
    ylabel('RI')
    colorbar
    
    figure
    contour(x1, x2, (reshape(exp(lp), size(x1)))');
    hold on
    scatter(xOrigFail(:,1), xOrigFail(:,2), 'rx')
    scatter(xOrigOkay(:,1), xOrigOkay(:,2), 'ro')
    
    scatter(xAddedFail(:,1), xAddedFail(:,2), 'bx')
    scatter(xAddedOkay(:,1), xAddedOkay(:,2), 'bo')
    
    title('Points added on probability contours')
    xlabel('gap ratio')
    ylabel('RI')
    colorbar
    
    sigE        = 0.2;
    T           = 0.05;
    Wx          = 1./sqrt(2*pi*(sigE^2 + ys2)) .* ...
        exp((-1/2)*((exp(lp) - T).^2./sigE^2 + ys2.^2));
    
    figure
    contour(x1, x2, (reshape(Wx.*ys2, size(x1)))');
    hold on
    scatter(xOrigFail(:,1), xOrigFail(:,2), 'rx')
    scatter(xOrigOkay(:,1), xOrigOkay(:,2), 'ro')
    
    scatter(xAddedFail(:,1), xAddedFail(:,2), 'bx')
    scatter(xAddedOkay(:,1), xAddedOkay(:,2), 'bo')
    
    title('Weighted variance')
    xlabel('gap ratio')
    ylabel('RI')
    colorbar
    
%     scatter(x(1:k, 1), x(1:k, 2), 'ro')
%     scatter(x((k+1):end, 1), x((k+1):end, 2), 'bx')

end

%% mean plot

function meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
    [ymu,~,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    
    figure
    
    surf(x1, x2, reshape(ymu, size(x1))');
    title('Mean plot')
    xlabel('gap ratio')
    ylabel('RI')
    zlim([-1.5,1.5])
    colorbar
end

%% variance plot

function varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
    [~,ys2,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    
    figure
    surf(x1, x2, reshape(ys2, size(x1))' );
    title('Variance plot')
    xlabel('gap ratio')
    ylabel('RI')
    zlim([0, 1])
    colorbar
end

%% MMSE function (GPML) (Kyprioti, McKay)
% considers only existing dataset

function [MMSE, xNext] = fn_mmse_gpml(hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk, xs)

    n = length(xs);
    
    [~,s2k,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk,...
        xs, ones(n,1));
    
    % Weighting
    
    sigE        = 0.2;
    T           = 0.05;
    Wx          = 1./sqrt(2*pi*(sigE^2 + s2k)) .* ...
        exp((-1/2)*((exp(lp) - T).^2./sigE^2 + s2k.^2));
        
    [MMSE, mmIdx]   = max(s2k.*Wx);
    xNext           = xs(mmIdx,:);
    
%     % find subset with probability of failure around 5%
%     Hk      = xs((exp(lp) <= 0.07) & (exp(lp) >= 0.03),:);
%     s2kHk   = s2k((exp(lp) <= 0.07) & (exp(lp) >= 0.03),:);
%     
%     if isempty(Hk)
%         [MMSE, mmIdx]   = max(s2k);
%         xNext           = xs(mmIdx,:);
%         disp('no test point found with p(f) around 5%')
%     else
%         [MMSE, mmIdx]   = max(s2kHk);
%         xNext           = Hk(mmIdx,:);
%     end

end

%% libSearch function
% search through the reserves to find closest point (Euclidean distance)

function [xFound, yFound, xReserve, yReserve] = libSearch(xNext, xReserve, yReserve)
    % normalize since dimensions have different scale
    xNormed     = xReserve./max(xReserve);
    xNextNorm   = xNext./max(xReserve);
    kIdx  = dsearchn(xNormed, xNextNorm);
    xFound = xReserve(kIdx,:);
    yFound = yReserve(kIdx);
    xReserve(kIdx,:) = [];
    yReserve(kIdx,:) = [];
end