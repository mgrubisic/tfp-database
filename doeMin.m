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

isolFull    = readtable('../pastRuns/random200withTfb.csv');
% limit to 20 points, e.g.
isolDat     = isolFull(1:20,:);
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
% x           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];
x           = [gapRatio, Ry];

% conference paper:
% x           = [gapRatio, T2Ratio, zeta, Ry];


y           = collapsed;
y(y==0)     = -1;

%% old GP
% intervals, mins, and max of variables
[e,f]       = size(x);

% try mean as constant
% meanfunc    = @meanConst; hyp.mean = 0;

% try ignoring the mean function
% meanfunc    = [];

% conference paper:
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

% conference paper:
hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
% hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);


%% GLM

% generalized linear model, logistic regression
% linFit      = fitglm(x,collapsed, 'Distribution','binomial','Intercept',false);
linFit      = fitglm(x, y, 'Intercept', false);
betaHatGLM  = linFit.Coefficients.Estimate;
hx          = linFit.Residuals.Raw;

% try mean as constant (although we defer mean to kriging)
meanfunc       = @meanZero; hyp.mean = [];
covfunc        = @covSEiso; hyp.cov = [0 0];
% gaussian likelihood
% likZ        = @likLogistic;
% infZ        = @infLaplace;
likfunc        = @likGauss; hyp.lik = -1;
inffunc        = @infGaussLik;
% hyp         = minimize(hyp, @gp, -3000, infZ, meanZ, covZ, likZ, x, y);
hyp         = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x, hx);
% C           = linFit.CoefficientCovariance;

%% Kyprioti paper
k           = length(x);

F           = x;
RMat        = covfunc(hyp.cov, x);
RInv        = inv(RMat);
betaHat     = inv(F'*RInv*F)*F'*RInv*y;
sig2Tilde   = (y - F*betaHat)'*(y - F*betaHat)/k;


mk          = zeros(k, 1);
sig2        = zeros(k, 1);
sig2n       = zeros(k, 1);

for i = 1:(k)
    r       = RMat(:,i);
    f       = F(i,:)';
    u       = F'*RInv*r - f;
    sig2n(i) = (1 - r'*RInv*r + u'*inv(F'*RInv*F)*u);
    testa(i) = r'*RInv*r;
    testb(i) = u'*inv(F'*RInv*F)*u;
    
    % predictive mean
    mk(i)       = f'*betaHat + r'*RInv*(y - F*betaHat);
    
    % predictive variance
    sig2(i)     = sig2Tilde*sig2n(i);
end


%% Kyprioti with xnew
xk = x;
% xnew = [0.04 8 1.1 0.12 1.8];
xnew    = [0.05, 2.0];
% xnew = [10, 10, 10, 10, 10];
Fnew = [xk;xnew];

cnew        = covfunc(hyp.cov, xk, xnew);
etaNew      = 1 - cnew'*RInv*cnew;
RNewInv    = [RInv+(RInv*cnew*cnew'*RInv)/etaNew -RInv*cnew/etaNew;
               -cnew'*RInv/etaNew 1/etaNew];

% CkNew       = [sig2Tilde cnew'; cnew Ck];
% CkNewInv    = [1 zeros(k,1)'; -inv(Ck)*cnew eye(k)]*...
%     [1/(sig2Tilde - cnew'*inv(Ck)*cnew) zeros(k,1)'; zeros(k,1) inv(Ck)]* ...
%     [1 -cnew'*inv(Ck); zeros(k,1) eye(k)];

sig2nNew    = zeros(k,1);

for i = 1:(k)
    cCur        = covfunc(hyp.cov, [xk; xnew], x(i,:));
    fnew        = F(i,:)';
    unew        = Fnew'*RNewInv*cCur - fnew;
    
    sig2nNew(i) = 1 + unew'*inv(Fnew'*RNewInv*Fnew)*unew - cCur'*RNewInv*cCur;
end

% Z           = y - F*betaHat;

 %% Picheny IMSEt using s2k from scratch (universal kriging)
% % K   = covfunc(hyp.cov, x);
% % if ys2 is unaffected by y observations, create dummy one w/ k+1 rows
% xk = x;
% xnew = [0.04 8 1.1 0.12 1.8];
% % xnew = [0.5, 10, 1.5, 0.20, 2.5];
% k           = length(xk);
% xSet        = [xnew; xk];
% 
% % from Schur's complement formula
% Ck          = covfunc(hyp.cov, xk);
% cnew        = covfunc(hyp.cov, xk, xnew);
% % cnew        = CkNewTemp(1:(end-1),end);
% sigVar      = Ck(1,1);
% CkNew       = [sigVar cnew'; cnew Ck];
% CkNewInv    = [1 zeros(k,1)'; -inv(Ck)*cnew eye(k)]*...
%     [1/(sigVar - cnew'*inv(Ck)*cnew) zeros(k,1)'; zeros(k,1) inv(Ck)]* ...
%     [1 -cnew'*inv(Ck); zeros(k,1) eye(k)];
% 
% 
% % basis function is linear, f(x) = x
% F           = xSet;
% s2k         = zeros(k,1);
% for i = 1:(k)
%     ki      = CkNew(i,i);
%     ci      = CkNew(:,i);
%     fi      = xSet(i,:)';
%     s2k(i)  = ki - ci'*CkNewInv*ci + (fi' - ci'*CkNewInv*F)*...
%         inv(F'*CkNewInv*F)*(fi'-ci'*CkNewInv*F)';
% %     test(i) = ki - ci'*CkNewInv*ci;
% %     test2(i)= (fi' - ci'*CkNewInv*F)*...
% %         inv(F'*CkNewInv*F)*(fi'-ci'*CkNewInv*F)';
% end
% 
% CkInv       = inv(Ck);
% s2kt        = zeros(k,1);
% Ft          = xk;
% mk          = zeros(k,1);
% betaHat = inv(Ft'*CkInv*Ft)*Ft'*CkInv*y;
% for i = 1:(k)
%     kt      = Ck(i,i);
%     ct      = Ck(:,i);
%     ft      = xk(i,:)';
%     s2kt(i)  = kt - ct'*CkInv*ct + (ft' - ct'*CkInv*Ft)*...
%         inv(Ft'*CkInv*Ft)*(ft'-ct'*CkInv*Ft)';
%     mk(i)   = ft'*betaHat + ct'*CkInv*(y - Ft*betaHat);
%     test(i) = kt - ct'*inv(Ck)*ct;
% end
% 
% 
% % training?
% % [nlZ, dnlZ] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[ymu,ys2,fmu,fs2,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x, ...
    ones(length(x), 1));
% 
% % IMSEt (Picheny)
% % PDFx        = exp(lp);
% mkx         = meanfunc(hyp.mean, x);
% T           = -0.90;
% 
% sigE        = 0.05*(max(y)- min(y));
% Wx          = 1./sqrt(2*pi*(sigE^2 + s2k)) .* ...
%     exp((-1/2)*((mk - T).^2./sigE^2 + s2k.^2));
% 
% IMSEt       = sum(s2k.*Wx);

