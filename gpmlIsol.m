%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	GPML Isolation

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	October 2020

% Description: 	Script attempts GP classification on isolator data, with
% improvements to plotting and using the entire dataset

% Open issues: 	(1) FITC sparse approx not done

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define data
clear; close all; clc;

isolDat     = readtable('../pastRuns/random200withTfb.csv');
g           = 386.4;

% scaling Sa(Tm) for damping, ASCE Ch. 17
zetaRef     = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
BmRef       = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0];
isolDat.Bm  = interp1(zetaRef, BmRef, isolDat.zetaM);

TfbRatio    = isolDat.Tfb./isolDat.Tm;
mu2Ratio    = isolDat.mu2./(isolDat.GMSTm./isolDat.Bm);
mu1Ratio    = isolDat.mu1./(isolDat.GMSTm./isolDat.Bm);
gapRatio    = isolDat.moatGap./(g.*(isolDat.GMSTm./isolDat.Bm).*isolDat.Tm.^2);
T2Ratio     = isolDat.T2./isolDat.Tm;
Ry          = isolDat.RI;
zeta        = isolDat.zetaM;

Tshort      = isolDat.S1/2.282;
% Tshort      = (isolDat.S1.*isolDat.S1Ampli)/(2.282);
TmRatio     = isolDat.Tm./Tshort;
% TmRatio     = isolDat.Tm./isolDat.Tfb;
% TmRatio     = isolDat.Tm.^2./(isolDat.GMSTm.*g./isolDat.R2);

% collapsed   = isolDat.impacted;

collapsed   = (isolDat.collapseDrift1 | isolDat.collapseDrift2) ...
    | isolDat.collapseDrift3;

collapsed   = double(collapsed);

% x should be all combinations of x1 and x2
% y should be their respective resulting impact boolean
% x           = [mu2Ratio, gapRatio, T2Ratio, zeta, Ry];
% x           = [gapRatio, T2Ratio, mu2Ratio, Ry];
% x           = [mu1Ratio, gapRatio, T2Ratio, zeta, Ry];
% x           = [mu2Ratio, T2Ratio, zeta, Ry];
x           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];

% conference paper:
% x           = [gapRatio, T2Ratio, zeta, Ry];


y           = collapsed;
y(y==0)     = -1;

%% GP model
% intervals, mins, and max of variables
[e,f]       = size(x);

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
% Logit regression
likfunc     = @likLogistic;
inffunc     = @infLaplace;

% conference paper:
% hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);

%% Plotting
close all
% Goal: for 3 values of T2 ratio, plot gapRatio vs. damping
% plotContour(constIdx, xIdx, yIdx, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)
% conference paper:
% plotContour(3, 1, 2, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

plotContour(1, 2, 3, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

% % Goal: set T2 ratio to median, fix three values damping ratios, plot marginal for
% % gap ratio (set x1, fix x3, plot x2)
% plotMarginalSlices(constIdx, xIdx, fixIdx, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)
% conference paper:
% plotMarginalSlices(4, 1, 2, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

plotMarginalSlices(5, 1, 2, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

%% Old cost regression
% Goal: get a design space of qualifying probs of failure, then pick one
% based on cost

% SaTm    = mean(isolDat.GMSTm);
% Tm      = mean(isolDat.Tm);
% Bm      = mean(isolDat.Bm);
% 
% minDm   = min(gapRatio)*g*SaTm/Bm*Tm^2;
% maxDm   = max(gapRatio)*g*SaTm/Bm*Tm^2;
% 
% % minDmCost   = 531/144*(90*12 + 2*minDm)^2;
% % maxDmCost   = 531/144*(90*12 + 2*maxDm)^2;
% minDmCost   = (0.20*417)/144*(90*12 + 2*minDm)^2;
% maxDmCost   = (0.20*417)/144*(90*12 + 2*maxDm)^2;
% dCostdGap   = (maxDmCost - minDmCost)/(maxDm - minDm);
% 
% minRyCost = 372030;
% maxRyCost = 205020;
% 
% dCostdRI = (maxRyCost - minRyCost)/(2.0 - 0.5);

% In general, some damping >10% is desired, and some T2Ratio < 1.2 is
% desired, so we "cost" T2Ratio and "reward" damping moderately
% [designSpace, designPoint, minidx] = minDesign(probDesired, steps, x, y, w, ...
%     hyp, meanfunc, covfunc, inffunc, likfunc)
% conference paper:
% weightVec   = [dCostdGap, 0.0, 0.0, dCostdRI, 0.0];

% lasso
% cost = f(moatGap, Tm, T2, zeta, Vs)
% [coefVec, coef0, lassoStruc]    = fnLasso(x);

% weightVec   = [dCostdGap, 0.0, 0.0, 0.0, dCostdRI, 0.0];
% [designSpace, designPoint, designSD] = minDesign(0.05, 5, x, y, ...
%     coefVec, coef0, hyp, meanfunc, covfunc, inffunc, likfunc);

%% Cost brute force: grid calculation

steelCoefs      = steelCost(isolDat);
probDesired     = 0.05;
gridRes         = 10;
[designSpace, designPoint, designFailureSD] = costGridCalc(probDesired, gridRes, x, y, ...
    steelCoefs, hyp, meanfunc, covfunc, inffunc, likfunc);
