%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Visualilzer

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	December 2020

% Description: 	Script plots stuff in MATLAB

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
A_S1        = isolDat.S1Ampli;
Tshort      = isolDat.S1/2.282;
TmRatio     = isolDat.Tm./Tshort;

maxDrift   = max([isolDat.driftMax1, isolDat.driftMax2, isolDat.driftMax3], [], 2);
impacted   = isolDat.impacted;

collapsed   = (isolDat.collapseDrift1 | isolDat.collapseDrift2) ...
    | isolDat.collapseDrift3;

%collapsed   = double(collapsed);

% gap ratio vs drift
figure
scatter(gapRatio(collapsed == 0), maxDrift(collapsed == 0), 75, '+')
hold on
grid on
scatter(gapRatio(collapsed == 1), maxDrift(collapsed == 1), 75, 'x')
legend('No collapse', 'Collapse', 'fontsize', 16)
set(gca,'FontSize', 18)
xlabel('Gap ratio', 'fontsize', 16, 'Interpreter', 'latex')
ylabel('Maximum drift', 'fontsize', 16, 'Interpreter', 'latex')

% T2 ratio vs drift
figure
scatter(T2Ratio(collapsed == 0), maxDrift(collapsed == 0), 75, '+')
hold on
grid on
scatter(T2Ratio(collapsed == 1), maxDrift(collapsed == 1), 75, 'x')
legend('No collapse', 'Collapse', 'fontsize', 16)
set(gca,'FontSize', 18)
xlabel('$T_2$ ratio', 'fontsize', 16, 'Interpreter', 'latex')
ylabel('Maximum drift', 'fontsize', 16, 'Interpreter', 'latex')

% damping ratio vs drift
figure
scatter(zeta(collapsed == 0), maxDrift(collapsed == 0), 75, '+')
hold on
grid on
scatter(zeta(collapsed == 1), maxDrift(collapsed == 1), 75, 'x')
legend('No collapse', 'Collapse', 'fontsize', 16)
set(gca,'FontSize', 18)
xlabel('$\zeta_M$', 'fontsize', 16, 'Interpreter', 'latex')
ylabel('Maximum drift', 'fontsize', 16, 'Interpreter', 'latex')

% Ry vs drift
figure
scatter(Ry(collapsed == 0), maxDrift(collapsed == 0), 75, '+')
hold on
grid on
scatter(Ry(collapsed == 1), maxDrift(collapsed == 1), 75, 'x')
legend('No collapse', 'Collapse', 'fontsize', 16)
set(gca,'FontSize', 18)
xlabel('$R_y$', 'fontsize', 16, 'Interpreter', 'latex')
ylabel('Maximum drift', 'fontsize', 16, 'Interpreter', 'latex')

% Ry vs drift (subset no impact)
figure
scatter(Ry(impacted == 0), maxDrift(impacted == 0), 75, '+', 'MarkerEdgeColor', [0 0.4470 0.7410])
grid on
set(gca,'FontSize', 18)
xlabel('$R_y$', 'fontsize', 16, 'Interpreter', 'latex')
ylabel('Maximum drift', 'fontsize', 16, 'Interpreter', 'latex')

% Tm ratio vs drift
figure
scatter(TmRatio(collapsed == 0), maxDrift(collapsed == 0), 75, '+')
hold on
grid on
scatter(TmRatio(collapsed == 1), maxDrift(collapsed == 1), 75, 'x')
legend('No collapse', 'Collapse', 'fontsize', 16)
set(gca,'FontSize', 18)
xlabel('$T_M$ ratio', 'fontsize', 16, 'Interpreter', 'latex')
ylabel('Maximum drift', 'fontsize', 16, 'Interpreter', 'latex')

figure
scatter(TmRatio, isolDat.moatGap)
xlabel('TmRatio')
ylabel('Moat gap')

figure
scatter(isolDat.Tm, isolDat.moatGap)
xlabel('Tm')
ylabel('Moat gap')