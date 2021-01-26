%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	MATLAB GP try

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	October 2020

% Description: 	Script attempts GP classification on isolator data

% Open issues: 	(1) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

isolDat     = readtable('../pastRuns/random200withTfb.csv');

TfbRatio    = isolDat.Tfb./isolDat.Tm;
mu2Ratio    = isolDat.mu2./isolDat.GMSTm;
gapRatio    = isolDat.moatGap./(isolDat.GMSTm.*isolDat.Tm.^2);
T2Ratio     = isolDat.T2./isolDat.Tm;
Ry          = isolDat.RI;
zeta        = isolDat.zetaM;
A_S1        = isolDat.S1Ampli;

allPis      = [TfbRatio, mu2Ratio, gapRatio, T2Ratio, Ry, zeta, A_S1];

collapsed   = (isolDat.collapseDrift1 | isolDat.collapseDrift2) ...
    | isolDat.collapseDrift3;

collapsed   = double(collapsed);

% x should be all combinations of x1 and x2
% y should be their respective resulting impact boolean
figure
x           = [A_S1, gapRatio];
y           = collapsed;
y(y==0)     = -1;
scatter(x(:,1), x(:,2), [], y)
xlabel('$X_1$','Interpreter','latex')
ylabel('$X_2$','Interpreter','latex')
colorbar



meanfunc    = @meanConst; hyp.mean = 0;
covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell ell sf]);
% Logit regression
likfunc     = @likLogistic;
inffunc     = @infLaplace;

% inducing points for sparse FITC approx
rangex1     = linspace(min(x(:,1)), max(x(:,1)), 10);
rangex2     = linspace(min(x(:,2)), max(x(:,2)), 10);
[u1,u2]     = meshgrid(rangex1, rangex2);
u           = [u1(:), u2(:)];
covfuncF    = {@apxSparse, {covfunc}, u};

hyp = minimize(hyp, @gp, -100, inffunc, meanfunc, covfuncF, likfunc, x, y);

% test set
[t1, t2]    = meshgrid(linspace(min(x(:,1)), max(x(:,1))), ...
    linspace(min(x(:,2)), max(x(:,2))));
t           = [t1(:) t2(:)];
n           = length(t);

[a, b, c, d, lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, ...
    x, y, t, ones(n, 1));

figure
scatter(x(:,1), x(:,2), [], y)
hold on
xlabel('$X_1$','Interpreter','latex')
ylabel('$X_2$','Interpreter','latex')
contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);
colorbar