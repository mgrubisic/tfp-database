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

TfbRatio    = isolDat.Tfb./isolDat.Tm;
mu2Ratio    = isolDat.mu2./isolDat.GMSTm;
gapRatio    = isolDat.moatGap./(g.*isolDat.GMSTm.*isolDat.Tm.^2);
T2Ratio     = isolDat.GMST2./isolDat.GMSTm;
Ry          = isolDat.RI;
zeta        = isolDat.zetaM;
A_S1        = isolDat.S1Ampli;

collapsed   = (isolDat.collapseDrift1 | isolDat.collapseDrift2) ...
    | isolDat.collapseDrift3;

collapsed   = double(collapsed);

% x should be all combinations of x1 and x2
% y should be their respective resulting impact boolean
% x           = [mu2Ratio, gapRatio, T2Ratio, zeta, Ry];
x           = [mu2Ratio, gapRatio, T2Ratio, Ry];
y           = collapsed;
y(y==0)     = -1;

% intervals, mins, and max of variables
[e,f]       = size(x);
minX        = round(min(x),1);
maxX        = round(max(x),1);
midX        = round(median(x),2);
stepX       = (maxX-minX)/50;

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

% % inducing points for sparse FITC approx
% nFITC       = 10;
% oneRange    = linspace(0, 1, nFITC);
% xRanges     = minX' + oneRange.*(maxX' - minX');
% uRanges     = cell(1, size(xRanges, 1));
% xuInput     = num2cell(xRanges, 2);
% 
% [uRanges{:}]    = ndgrid(xuInput{:});
% u               = [uRanges{:}];
% covfuncF    = {@apxSparse, {covfunc}, u};

% rangex1     = linspace(min(x(:,1)), max(x(:,1)), 10);
% rangex2     = linspace(min(x(:,2)), max(x(:,2)), 10);
% [u1,u2]     = meshgrid(rangex1, rangex2);
% u           = [u1(:), u2(:)];
% covfuncF    = {@apxSparse, {covfunc}, u};
% 
% hyp = minimize(hyp, @gp, -100, inffunc, meanfunc, covfuncF, likfunc, x, y);

% Goal: for 3 values of mu2Ratio, plot gapRatio vs. T2Ratio
% plotContour(constIdx, xIdx, yIdx, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)
plotContour(1, 2, 3, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

% % Goal: set mu2 ratio to median, fix three values T2 ratios, plot marginal for
% % gap ratio (set x1, fix x3, plot x2)
% plotMarginalSlices(constIdx, xIdx, fixIdx, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)
plotMarginalSlices(1, 2, 3, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

% for i = 1:length(midX)
%     xs(:,i) = transpose(minX(i):stepX(i):maxX(i));
% end