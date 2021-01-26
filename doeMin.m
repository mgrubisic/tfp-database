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

isolDat     = readtable('../pastRuns/random200withTfb.csv');
% limit to 20 points, e.g.
isolDat     = isolDat(1:20,:);
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
TmRatio     = isolDat.Tm./Tshort;

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

% intervals, mins, and max of variables
[e,f]       = size(x);

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
% Logit regression
likfunc     = @likLogistic;
inffunc     = @infLaplace;

% conference paper:
hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
% hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);

%%
% % training?
% [nlZ, dnlZ] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
% [ymu,ys2,fmu,fs2,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x, ...
%     ones(length(x), 1));
% 
% % IMSEt (Picheny)
% PDFx        = exp(lp);
% T           = -0.90;
% 
% sigE        = 0.05*(max(y)- min(y));
% Wx          = 1./sqrt(2*pi*(sigE^2 + ys2)) .* ...
%     exp((-1/2)*((ymu - T).^2./sigE^2 + ys2.^2));
% 
% IMSEt       = sum(ys2.*Wx.*PDFx);
