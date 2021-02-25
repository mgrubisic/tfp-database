%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Isolation cost, LASSO regression

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	February 2021

% Description: 	Script performs LASSO regression on cost of isolation data
% set to assist in optimization and inverse design

% Open issues: 	(1)

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
mu1Ratio    = isolDat.mu1./(isolDat.GMSTm./isolDat.Bm);
gapRatio    = isolDat.moatGap./(g.*(isolDat.GMSTm./isolDat.Bm).*isolDat.Tm.^2);
T2Ratio     = isolDat.T2./isolDat.Tm;
Ry          = isolDat.RI;
zeta        = isolDat.zetaM;

Tshort      = isolDat.S1/2.282;
TmRatio     = isolDat.Tm./Tshort;

%%
% Cost lasso

isolDat.beamWeight  = str2double(extractAfter(isolDat.beam,'X'));
isolDat.roofWeight  = str2double(extractAfter(isolDat.roofBeam,'X'));
isolDat.colWeight   = str2double(extractAfter(isolDat.col,'X'));

% 624 ft of columns, 720 ft of beam, 360 ft of roof
isolDat.bldgWeight  = 624*isolDat.colWeight + 720*isolDat.beamWeight + ...
    360*isolDat.roofWeight;

% $1.25/lb of steel
isolDat.steelCost   = 1.25*isolDat.bldgWeight;

isolDat.kM          = (1/g)*(2*pi./isolDat.Tm).^2;
% 2 layers of frame per direction
Ws                  = 2227.5;
W                   = 3037.5;
isolDat.Dm          = g*1.017.*isolDat.Tm./(4*pi^2.*isolDat.Bm);
isolDat.Vb          = (isolDat.Dm.*isolDat.kM*2227.5)/2;
isolDat.Vst         = (isolDat.Vb.*(Ws/W).^(1 - 2.5*isolDat.zetaM));
isolDat.Vs          = isolDat.Vst./isolDat.RI;

% Hogan: land cost is about 20% of the development cost ($1110/sf)
landCostPerSqft     = 0.2*1110;
isolDat.landCost    = landCostPerSqft/144*(90*12 + 2*isolDat.moatGap).^2;

YCost               = isolDat.landCost + isolDat.steelCost;
% XCost               = [gapRatio, TmRatio, Ry];
% XCost               = [isolDat.moatGap, isolDat.Tm, Ry];
% XCost               = [gapRatio, TmRatio, T2Ratio, zeta, Ry];
XCost               = [isolDat.moatGap, isolDat.Tm, isolDat.T2, zeta, isolDat.Vs];

% Split the data into training and test sets
n           = length(YCost);
c           = cvpartition(n,'HoldOut',0.3);
idxTrain    = training(c,1);
idxTest     = ~idxTrain;
XTrain      = XCost(idxTrain,:);
yTrain      = YCost(idxTrain);
XTest       = XCost(idxTest,:);
yTest       = YCost(idxTest);

% Construct the lasso fit using 10-fold cross-validation
% Use the largest Lambda value such that the mean squared error (MSE) 
% is within one standard error of the minimum MSE.
[B,FitInfo]     = lasso(XTrain,yTrain,'CV',10);
idxLambda1SE    = FitInfo.Index1SE;
coef            = B(:,idxLambda1SE);
coef0           = FitInfo.Intercept(idxLambda1SE);

% plot
yhat            = XTest*coef + coef0;

figure
hold on
scatter(yTest,yhat)
plot(yTest,yTest)
xlabel('Actual system costs')
ylabel('Predicted system costs')
% axis equal
hold off

lassoPlot(B,FitInfo,'PlotType','CV');
legend('show')

%%
% Use simple least squares regression to find steel cost as function of Vs
linMdl      = fitlm(isolDat.Vs, isolDat.steelCost);
b0          = linMdl.Coefficients.Estimate(1);
b1          = linMdl.Coefficients.Estimate(2);
predCost    = b1*isolDat.Vs + b0;

figure
hold on
scatter(isolDat.steelCost, predCost)
plot(isolDat.steelCost, isolDat.steelCost)
xlabel('Actual steel costs')
ylabel('Predicted steel costs')