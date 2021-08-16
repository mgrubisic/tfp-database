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
% isolFull    = readtable('../pastRuns/random200withTfb.csv');
isolFull    = readtable('../pastRuns/random600.csv');

% scaling Sa(Tm) for damping, ASCE Ch. 17
g           = 386.4;
zetaRef     = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
BmRef       = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0];

isolFull.Tshort      = (isolFull.S1)/2.282;
% isolFull.Tshort      = (isolFull.S1.*isolFull.S1Ampli)/2.282;

% Consider filtering out cases where the Tshort is smaller
% to prevent distortion of TmRatio
% isolFull    = isolFull(isolFull.Tshort >= 0.4,:);

isolFull.Bm  = interp1(zetaRef, BmRef, isolFull.zetaM);

TfbRatio    = isolFull.Tfb./isolFull.Tm;
mu2Ratio    = isolFull.mu2./(isolFull.GMSTm./isolFull.Bm);
mu1Ratio    = isolFull.mu1./(isolFull.GMSTm./isolFull.Bm);
% gapRatio    = isolFull.moatGap./(g.*(isolFull.GMSTm./isolFull.Bm).*isolFull.Tm.^2);
gapRatio    = (isolFull.moatGap*4*pi^2)./(g.*(isolFull.GMSTm./isolFull.Bm).*isolFull.Tm.^2);
T2Ratio     = isolFull.T2./isolFull.Tm;
T1Ratio     = isolFull.T1./isolFull.Tm;
Ry          = isolFull.RI;
zeta        = isolFull.zetaM;

TmRatio     = isolFull.Tm./isolFull.Tshort;
% TmRatio     = isolFull.Tm./(1.107/2.2815);

collapsed   = (isolFull.collapseDrift1 | isolFull.collapseDrift2) ...
    | isolFull.collapseDrift3;

collapsed   = double(collapsed);
collapsed(collapsed==0)   = -1;

maxDrift    = max([isolFull.driftMax1, isolFull.driftMax2, isolFull.driftMax3], ...
    [], 2);

%% Classification GP

xFull       = [gapRatio, TmRatio];
yFull       = collapsed;
% xFull       = [gapRatio, TmRatio, zeta, Ry];
% xFull       = [gapRatio, TmRatio, mu1Ratio, zeta, Ry];

% limit to 90th quantile of gap ratios
xFull       = xFull(gapRatio <= quantile(gapRatio, 0.9),:);
yFull       = yFull(gapRatio <= quantile(gapRatio, 0.9),:);

% limit to ninit points
ninit       = 20;
randSet     = randsample(height(xFull), ninit);
x           = xFull(randSet,:);
xReserve    = xFull(setdiff(1:height(xFull), randSet), :);

y           = yFull(randSet);
yReserve    = yFull(setdiff(1:height(xFull), randSet));

k           = length(x);

[~,f]       = size(x);

% try mean as constant
meanfunc    = @meanConst; hyp.mean = 0;

% try ignoring the mean function
% meanfunc    = [];

% conference paper:
% try mean as affine function
% meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [zeros(1,f) 0]';

% try mean as linear function
% meanfunc    = @meanLinear; hyp.mean = [zeros(1,f)]';

% poly?
% meanfunc    = {@meanPoly,2}; hyp.mean = [zeros(1,2*f)]';

covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell*ones(1,f) sf]);
% covfunc     = @covSEiso; hyp.cov = [0 0];

% Logit regression or probit regression
likfunc     = @likErf;
inffunc     = @infLaplace;

% conference paper:
hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
% hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);

%% sequentially add points (option to update hyp) (classification)
% Run optimization
lPoints = 10;

% Domain generation
close all;
steps       = 15;
minX        = round(min(xFull),2);
maxX        = round(max(xFull),2);

gridVec     = cell(1, f);

for j = 1:f
    gridVec{j} = linspace(minX(j), maxX(j), steps);
end

xs   = transpose(combvec(gridVec{:}));

xsOG    = xs;

[x1, x2]  = meshgrid(gridVec{1}',gridVec{2}');

if (f == 2)
    varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
    meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
    evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2, k)
end

% covTracker = zeros(lPoints, f);
% meanTracker = zeros(lPoints, f);
xSuggest    = zeros(lPoints, f);

warning('off')
for i = 1:lPoints
    [~, xNext]   = fn_mmse_gpml(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    xSuggest(i,:) = xNext;
    
    [xFound, yFound, xReserve, yReserve] = libSearch(xNext, xReserve, yReserve);
    
    x       = [x; xFound];
    y       = [y; yFound];
     
    % redo GP?
    [e,f]       = size(x);
    meanfunc    = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [zeros(1,f) 0]';
    covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell*ones(1,f) sf]);
    likfunc     = @likErf;
    inffunc     = @infLaplace;
    hyp = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x, y);
    covTracker(i,:) = hyp.cov;
    meanTracker(i,:) = hyp.mean;
    
	% remesh to improve resolution of search
    xs      = remesh(xFull, f, xFound, steps);
    
%     % write DoE point to csv file
%     writeInput(filepath, xNext)
end
warning('on')

% final plots
if (f == 2)
    varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xsOG, x1, x2)
    meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xsOG, x1, x2)
    evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xsOG, x1, x2, k)
end


%% write input file

function writeInput(filename, doePoint)

    % randomly generate values within bounds
    S1 = (1.3 - 0.8)*rand + 0.8;
    S1Ampli = (2.25 - 1.0)*rand + 1.0;
    mu1 = (0.05 - 0.01)*rand + 0.01;
    R1 = (45 - 15)*rand + 15;
    
    % write csv
    fid = fopen(filename,'w');
    fprintf(fid, 'variable,value\n');
    fprintf(fid, 'S1,%5.4f\n', S1);
    fprintf(fid, 'S1Ampli,%5.4f\n', S1Ampli);
    fprintf(fid, 'mu1,%5.4f\n', mu1);
    fprintf(fid, 'R1,%5.4f\n', R1);
    formatSpec = 'gapRatio,%5.4f\nTmRatio,%5.4f\nzetaM,%5.4f\nRI,%5.4f';
    fprintf(fid, formatSpec, doePoint(1,:));
    fclose(fid);
end
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
    [ymu,ys2,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, ...
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
%     [C, h] = contour(x1, x2, (reshape(exp(lp), size(x1)))', [0.0:0.05:1.0]);
%     v = [0.05, 0.5, 0.95];
    [C, h] = contour(x1, x2, (reshape(exp(lp), size(x1)))', [0.0:0.01:0.1]);
    v = [0.01, 0.05, 0.1];
    clabel(C,h,v)
    hold on
    scatter(xOrigFail(:,1), xOrigFail(:,2), 'rx')
    scatter(xOrigOkay(:,1), xOrigOkay(:,2), 'ro')
    
    scatter(xAddedFail(:,1), xAddedFail(:,2), 'bx')
    scatter(xAddedOkay(:,1), xAddedOkay(:,2), 'bo')
    
    title('Points added on probability contours')
    xlabel('gap ratio')
    ylabel('RI')
    colorbar
    
    % Weighting
    
    sigE        = 0.1;
    T           = -1;
    Wx          = 1./sqrt(2*pi*(sigE^2 + ys2)) .* ...
        exp((-1/2)*((ymu - T).^2./sigE^2 + ys2.^2));
    
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
    [ymu,~,fmu,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    
    figure
    
    surf(x1, x2, reshape(ymu, size(x1))');
    title('Mean plot')
    xlabel('gap ratio')
    ylabel('RI')
    zlim([-1.5,1.5])
    colorbar
    
    figure
    
    surf(x1, x2, reshape(fmu, size(x1))');
    title('Latent plot')
    xlabel('gap ratio')
    ylabel('RI')
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
    
    [ymu,s2k,fmu,fs2,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk,...
        xs, ones(n,1));
    
    % Weighting
    sigE        = 0.1;  % recommended at 5% of range

%     Wx          = 1./sqrt(2*pi*(sigE^2 + s2k)) .* ...
%         exp((-1/2)*((exp(lp) - T).^2./sigE^2 + s2k.^2));

    % output mean and variance formulation
    T           = -1;
    pcol        = exp(lp);
    region5     = (pcol <= 0.055 & pcol >= 0.045);
    var5        = s2k(region5);
    topThresh   = ymu(region5) + sqrt(var5);
    
    region10    = (pcol >= 0.10);
    thresh10    = min(ymu(region10));
    
    if max(topThresh) <= thresh10
        disp('The 5% region is within +1 std of 10%.')
        disp(max(topThresh))
    end

    Wx          = 1./sqrt(2*pi*(sigE^2 + s2k)) .* ...
        exp((-1/2)*((ymu - T).^2./sigE^2 + s2k.^2));

    [MMSE, mmIdx]   = max(s2k.*Wx);
    xNext           = xs(mmIdx,:);
   
    % % Latent formulation
    % % Latent variable is the mean passed through probit
    % pcol        = exp(lp);
    % region5     = (pcol <= 0.055 & pcol >= 0.045);
    % var5        = fs2(region5);
    % topThresh   = fmu(region5) + sqrt(var5);
    
    % thresh10    = norminv(0.1);
    
    % if max(topThresh) <= thresh10
    %     disp('The 5% region is within +1 std of 10%.')
    %     disp(max(topThresh))
    % end
    
    % T           = norminv(0.05);

    % Wx          = 1./sqrt(2*pi*(sigE^2 + fs2)) .* ...
    %     exp((-1/2)*((fmu - T).^2./sigE^2 + fs2.^2));
    
    % [MMSE, mmIdx]   = max(fs2.*Wx);
    % xNext           = xs(mmIdx,:);
    
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