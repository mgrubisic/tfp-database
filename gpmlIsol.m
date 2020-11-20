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

v1  = [minX(1) midX(1) maxX(1)];

figure
for i = 1:length(v1)
    
    subplot(1,3,i)
    
    [t1, t2] = meshgrid(minX(2):stepX(2):maxX(2),minX(3):stepX(3):maxX(3));
    t = [t1(:) t2(:)]; n = length(t);
    t = [ones(n,1)*v1(i) t ones(n,1)*midX(4:f)];                  %creates points at which the model will be evaluated
    [a,b,c,d,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1)); %a is expected value, b is sd of expected value, lp is the probabilities used to print the contour curves

    % get points close to the desired v1's
    xPlot = x(( (x(:,1) < v1(i)+2*stepX(1)) & (x(:,1) > v1(i)-2*stepX(1)) ),:);
    yPlot = y(( (x(:,1) < v1(i)+2*stepX(1)) & (x(:,1) > v1(i)-2*stepX(1)) ),:);
    
    % plot data
    collapsedIdx   = yPlot == 1;
    notIdx = yPlot == -1;
    
    plot(xPlot(collapsedIdx,2), xPlot(collapsedIdx,3), 'r+'); hold on;
    plot(xPlot(notIdx,2), xPlot(notIdx,3), 'b+');
    
    xlabel('Gap ratio','Interpreter','latex')
    ylabel('$T_2$ ratio','Interpreter','latex')
    contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);
    
end
colorbar
sgtitle('Varying $\mu_2$ ratio', 'Interpreter', 'LaTeX')

% Goal: set mu2 ratio to median, fix three values T2 ratios, plot marginal for
% gap ratio (set x1, fix x3, plot x2)
t = [t1(:) t2(:)]; n = length(t);
t = [ones(n,1)*v1(2) t ones(n,1)*midX(4:f)];
[a,b,c,d,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));

v3  = [midX(3)-10*stepX(3) midX(3) midX(3)+10*stepX(3)];
gl = length(t1);

hold off
figure

for i = 1:length(v3)
    subplot(length(v3),1,i)
    
%     num = round((v3(i)-minX(3))/stepX(3));
%     q = gl*num + 1;
%     v = q + gl - 1 ;
%     pts = t(q:v,2); %y values are changing
    
    [pts, idc] = unique(t(:,2));
    if (i == 1)
        idc = idc + 7;
    elseif (i == 2)
        idc = idc + 17;
    else
        idc = idc + 27;
    end
    
    f = [a(idc)+2*sqrt(b(idc)) ; flip(a(idc)-2*sqrt(b(idc)), 1)];
    fill([pts;flip(pts,1)], f, [7 7 7]/8)
    hold on;
    plot(pts, a(idc))
    
    % get points close to the desired v1's
    xPlot = x(( (x(:,1) < v1(2)+2*stepX(1)) & (x(:,1) > v1(2)-2*stepX(1)) ),:);
    yPlot = y(( (x(:,1) < v1(2)+2*stepX(1)) & (x(:,1) > v1(2)-2*stepX(1)) ),:);
    
    xPlot = xPlot(( (xPlot(:,3) < v3(i)+5*stepX(3)) & (xPlot(:,3) > v3(i)-5*stepX(3)) ),:);
    yPlot = yPlot(( (xPlot(:,3) < v3(i)+5*stepX(3)) & (xPlot(:,3) > v3(i)-5*stepX(3)) ),:);
    
    % plot data
    collapsedIdx   = yPlot == 1;
    notIdx = yPlot == -1;
    
    plot(xPlot(collapsedIdx,2), yPlot(collapsedIdx), 'r+'); hold on;
    plot(xPlot(notIdx,2), yPlot(notIdx), 'b+');
    
    xlabel('Gap ratio','Interpreter','latex')
    ylabel('Impacted?','Interpreter','latex')
    
end

sgtitle('Gap ratio marginals for varying $T_2$ ratios', 'Interpreter', 'LaTeX')

% % test set
% [t1, t2]    = meshgrid(linspace(min(x(:,1)), max(x(:,1))), ...
%     linspace(min(x(:,2)), max(x(:,2))));
% t           = [t1(:) t2(:)];
% n           = length(t);
% 
% [a, b, c, d, lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, ...
%     x, y, t, ones(n, 1));
% 
% figure
% scatter(x(:,1), x(:,2), [], y)
% hold on
% xlabel('$X_1$','Interpreter','latex')
% ylabel('$X_2$','Interpreter','latex')
% contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);
% colorbar

for i = 1:length(midX)
    xs(:,i) = transpose(minX(i):stepX(i):maxX(i));
end