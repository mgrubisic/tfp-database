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

% limit to 20 points, e.g.
randSet     = randsample(height(isolFull), 20);
isolDat     = isolFull(randSet,:);
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
% x           = [gapRatio, Ry];

% conference paper:
% x           = [gapRatio, T2Ratio, zeta, Ry];

maxDrift    = max([isolDat.driftMax1, isolDat.driftMax2, isolDat.driftMax3], ...
    [], 2);

%% old GP
% intervals, mins, and max of variables

x           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];
% x           = gapRatio
% x           = [gapRatio, Ry];
y           = collapsed;
y(y==0)     = -1;

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
% covfunc        = @covSEiso; hyp.cov = [0 0];

% Logit regression
likfunc     = @likLogistic;
inffunc     = @infLaplace;

% conference paper:
hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
% hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);

% xs      = linspace(min(gapRatio), max(gapRatio), 61)';
% [mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
% 
% f = [mu+sqrt(s2); flip(mu-sqrt(s2),1)];
% fill([xs; flip(xs,1)], f, [7 7 7]/8)
% hold on; plot(xs, mu); plot(x, y, '+')

%% Regression try GP
% x           = [gapRatio, Ry];
x           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];
k           = length(x);
y           = maxDrift;

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
% meanfunc    = {@meanPoly,3}; hyp.mean = [zeros(1,3*f)]';

% covfunc     = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell*ones(1,f) sf]);
covfunc        = @covSEiso; hyp.cov = [0 0];

% Logit regression
likfunc     = @likGauss; hyp.lik = log(0.1);
inffunc     = @infGaussLik;

% conference paper:
% hyp = minimize(hyp, @gp, -3000, inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);

%% Plot
% xs      = linspace(min(gapRatio), max(gapRatio), 61)';
% [mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
% 
% f = [mu+sqrt(s2); flip(mu-sqrt(s2),1)];
% fill([xs; flip(xs,1)], f, [7 7 7]/8)
% hold on; plot(xs, mu); plot(x, y, '+')


%% sequentially add points (option to update hyp)
% Run optimization
options = optimset('MaxFunEvals',1000, 'GradObj', 'off'); %maximum 1000 iterations, gradient of the function not provided
firstTry    = mean(x);
lb  = min(x);
ub  = max(x);

lPoints = 3;

% plotting presets
close all;
minX        = round(min(x),2);
maxX        = round(max(x),2);
stepX       = (maxX-minX)/10;

gridVec     = cell(1, f);

for j = 1:f
    gridVec{j} = minX(j):stepX(j):maxX(j);
end

xs   = transpose(combvec(gridVec{:}));
n   = length(xs);

[x1, x2]  = meshgrid(gridVec{1}',gridVec{2}');

varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2, k)

covTracker = zeros(lPoints,2);
meanTracker = zeros(lPoints, 2);

warning('off')
for i = 1:lPoints
%     xNext   = fminsearchbnd(@fn_imse, firstTry, lb, ub, options, hyp, covfunc, x);
%     xNext   = fminsearchbnd(@fn_imse_gpml, firstTry, lb, ub, options, hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
    [~, xNext]   = fn_mmse_gpml(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
    disp(xNext)
    x       = [x; xNext];
    y       = camelback(x);
    
    % redo GP?
    [e,f]       = size(x);
%     meanfunc    = @meanLinear; hyp.mean = [zeros(1,f)]';
    meanfunc    = @meanConst; hyp.mean = 0;
    covfunc        = @covSEiso; hyp.cov = [0 0];
    likfunc     = @likGauss; hyp.lik = log(0.1);
    inffunc     = @infGaussLik;
    hyp = minimize(hyp, @gp, -200, inffunc, meanfunc, covfunc, likfunc, x, y);
    covTracker(i,:) = hyp.cov;
    meanTracker(i,:) = hyp.mean;
    
%     varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
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
varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)

evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2, k)

%% Surf plot
% close all;
% x1    = linspace(min(gapRatio),max(gapRatio),32);
% x2    = linspace(min(Ry),max(Ry),32);
% 
% xs      = combvec(x1, x2)';
% 
% [x1s, x2s]  = meshgrid(x1,x2);
% 
% [ymu,ys2,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
% 
% figure
% surf(x1s, x2s, reshape(ys2, size(x1s)));
% colorbar
% 
% figure
% surf(x1s, x2s, reshape(ymu, size(x1s)));
% colorbar

 %% Kyprioti paper
% k           = length(x);
% 
% F           = x;    % universal kriging, using linear
% % F           = ones(k,1);    % Echard, ordinary kriging, using constant
% RMat        = covfunc(hyp.cov, x);
% RInv        = RMat\eye(k);
% % RInv        = inv(RMat);
% betaHat     = (F'*RInv*F)\(F'*RInv*y);
% % betaHat     = inv(F'*RInv*F)*F'*RInv*y;
% % sig2Tilde   = (y - F*betaHat)'*RInv*(y - F*betaHat)/k; % Echard & Schobi
% sig2Tilde   = (y - F*betaHat)'*(y - F*betaHat)/k; % Kyprioti
% 
% mk          = zeros(k, 1);
% sig2        = zeros(k, 1);
% sig2n       = zeros(k, 1);
% 
% for i = 1:(k)
%     r       = RMat(:,i);
%     f       = F(i,:)';
%     u       = F'*RInv*r - f;
%     sig2n(i) = (1 - r'*RInv*r + u'*inv(F'*RInv*F)*u);
% %     sig2n(i) = (1 - r'*RInv*r + u'*(F'*RInv*F)\u);
%     testa(i) = r'*RInv*r;
%     testb(i) = u'*inv(F'*RInv*F)*u;
% %     testb(i) = u'*(F'*RInv*F)\u;
%     
%     % predictive mean
%     mk(i)       = f'*betaHat + r'*RInv*(y - F*betaHat);
%     
%     % predictive variance
%     sig2(i)     = sig2Tilde*sig2n(i);
% end
% 
% %% Kyprioti with xnew
% xk = x;
% % xnew = [0.04 8 1.1 0.12 1.8];
% xnew    = [0.05, 2.0];
% % xnew = [10, 10, 10, 10, 10];
% Fnew = [xk;xnew];
% 
% cnew        = covfunc(hyp.cov, xk, xnew);
% etaNew      = 1 - cnew'*RInv*cnew;
% RNewInv    = [RInv+(RInv*cnew*cnew'*RInv)/etaNew -RInv*cnew/etaNew;
%                -cnew'*RInv/etaNew 1/etaNew];
% 
% % CkNew       = [sig2Tilde cnew'; cnew Ck];
% % CkNewInv    = [1 zeros(k,1)'; -inv(Ck)*cnew eye(k)]*...
% %     [1/(sig2Tilde - cnew'*inv(Ck)*cnew) zeros(k,1)'; zeros(k,1) inv(Ck)]* ...
% %     [1 -cnew'*inv(Ck); zeros(k,1) eye(k)];
% 
% sig2nNew    = zeros(k,1);
% 
% for i = 1:(k)
%     cCur        = covfunc(hyp.cov, [xk; xnew], x(i,:));
%     fnew        = F(i,:)';
%     unew        = Fnew'*RNewInv*cCur - fnew;
%     
%     sig2nNew(i) = 1 + unew'*inv(Fnew'*RNewInv*Fnew)*unew - cCur'*RNewInv*cCur;
% end
% 
% % Z           = y - F*betaHat;

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
% [ymu,ys2,fmu,fs2,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x, y);
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

%% evolution plot

function evolplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2, k)
    [~,ys2,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    
    figure
    contour(x1, x2, reshape(ys2, size(x1)));
    hold on
    scatter(x(1:k, 1), x(1:k, 2), 'ro')
    scatter(x((k+1):end, 1), x((k+1):end, 2), 'bx')
    title('Points added on variance contours')
    xlabel('gap ratio')
    ylabel('RI')
    colorbar
end

%% variance plot

function varplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
    [~,ys2,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    
    figure
    surf(x1, x2, reshape(ys2, size(x1)));
    title('Variance plot')
    xlabel('gap ratio')
    ylabel('RI')
    colorbar
end

%% mean plot

function meanplot(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs, x1, x2)
    [ymu,~,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
    
    figure
    surf(x1, x2, reshape(ymu, size(x1)));
    title('Mean plot')
    xlabel('gap ratio')
    ylabel('RI')
    colorbar
end

%% MMSE function (GPML) (Kyprioti, McKay)
% considers only existing dataset

function [MMSE, xNext] = fn_mmse_gpml(hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk)
    % domain generation
    ulin    = linspace(-1,1,10);
    vlin    = linspace(-1,1,10);

    xs      = combvec(ulin, vlin)';
    
    [~,s2k,~,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk,...
        xs);
    
    [MMSE, mmIdx]   = max(s2k);
    xNext           = xs(mmIdx,:);
end