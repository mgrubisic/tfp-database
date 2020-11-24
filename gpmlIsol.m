%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	GPML Isolation

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	October 2020

% Description: 	Script attempts GP classification on isolator data, with
% improvements to plotting and using the entire dataset

% Open issues: 	(1) FITC sparse approx not done

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

isolDat     = readtable('../pastRuns/random200withTfb.csv');
g           = 386.4;

% scaling Sa(Tm) for damping, ASCE Ch. 17
zetaRef     = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
BmRef       = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0];
isolDat.Bm  = interp1(zetaRef, BmRef, isolDat.zetaM);

TfbRatio    = isolDat.Tfb./isolDat.Tm;
mu2Ratio    = isolDat.mu2./(isolDat.GMSTm./isolDat.Bm);
gapRatio    = isolDat.moatGap./(g.*(isolDat.GMSTm./isolDat.Bm).*isolDat.Tm.^2);
T2Ratio     = isolDat.T2./isolDat.Tm;
Ry          = isolDat.RI;
zeta        = isolDat.zetaM;
A_S1        = isolDat.S1Ampli;

% collapsed   = isolDat.impacted;

collapsed   = (isolDat.collapseDrift1 | isolDat.collapseDrift2) ...
    | isolDat.collapseDrift3;

collapsed   = double(collapsed);

% x should be all combinations of x1 and x2
% y should be their respective resulting impact boolean
x           = [mu2Ratio, gapRatio, T2Ratio, zeta, Ry];
% x           = [mu2Ratio, gapRatio, T2Ratio, Ry];
y           = collapsed;
y(y==0)     = -1;

% intervals, mins, and max of variables
[e,f]       = size(x);

% try mean as constant
% meanfunc    = @meanConst; hyp.mean = 0;

% try ignoring the mean function
% meanfunc    = [];

% try mean as affine function
% meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = [zeros(1,f) 0]';

% try mean as linear function
meanfunc    = @meanLinear; hyp.mean = [zeros(1,f)]';

% poly?
% meanfunc    = {@meanPoly,2}; hyp.mean = [zeros(1,2*f)]';

covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell*ones(1,f) sf]);
% Logit regression
likfunc     = @likLogistic;
inffunc     = @infLaplace;

hyp = minimize(hyp, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, x, y);


% Goal: for 3 values of mu2Ratio, plot gapRatio vs. T2Ratio
% plotContour(constIdx, xIdx, yIdx, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)
plotContour(1, 2, 3, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)
% plotContour(2, 1, 3, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

% % Goal: set mu2 ratio to median, fix three values T2 ratios, plot marginal for
% % gap ratio (set x1, fix x3, plot x2)
% plotMarginalSlices(constIdx, xIdx, fixIdx, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)
plotMarginalSlices(1, 2, 3, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

%%
% Goal: get a design space of qualifying probs of failure over 2 variables
% [designSpace, boundLine] = getDesignSpace(varX, varY, probDesired, probTol, x, y, hyp, meanfunc, covfunc, inffunc, likfunc)
[design, boundary] = getDesignSpace(2, 3, 0.1, 0.02, x, y, hyp, meanfunc, covfunc, inffunc, likfunc);