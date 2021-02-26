%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	UQLab Kriging Isolation

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	October 2020

% Description: 	Script attempts GP classification on isolator data, with
% improvements to plotting and using the entire dataset using UQLab

% Open issues: 	(1)

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
TmRatio     = isolDat.Tm./Tshort;


collapsed   = (isolDat.collapseDrift1 | isolDat.collapseDrift2) ...
    | isolDat.collapseDrift3;

collapsed   = double(collapsed);


x           = [gapRatio, TmRatio, T2Ratio, zeta, Ry];
y           = collapsed;
y(y==0)     = -1;

%%

% D e f i n e a K r i g i n g metamodel
MetaOpts.Type = 'Metamodel';
MetaOpts.MetaType = 'Kriging';
MetaOpts.ExpDesign.Sampling = 'User';
MetaOpts.ExpDesign.X = x;
MetaOpts.ExpDesign.Y = y;
MetaOpts.Trend.Type = 'linear';
MetaOpts.Corr.Type  = 'separable';
MetaOpts.Corr.Family = 'Gaussian';

% C r e a t e t h e K r i g i n g metamodel
myKriging = uq_createModel(MetaOpts);
