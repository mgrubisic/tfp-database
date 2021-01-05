%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	MLE Lognormal fit (fragility curve)

% Created by: 	Ya-Heng Yang
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script fits a lognormal distribution using a maximum
% likelihood method (with binomial likelihood) to the collapse data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
clc

% example data: IM levels, number of analyses, and number of collapses
IM = [1, 1.5, 2.0]; 
num_collapse = [2, 25, 43]; 
num_gms = [54, 54, 54];

% estimate fragility function using MLE method
[theta_hat_mle, beta_hat_mle] = fn_mle(IM, num_gms, num_collapse);

% compute fragility functions using estimated parameters
xmax = 4;
x_vals = 0.01:0.01:xmax;
p_collapse_mle_near = normcdf((log(x_vals/theta_hat_mle))/beta_hat_mle); % compute fragility function using equation 1 and estimated parameters

%% plot resulting fragility functions
figure
plot(IM,num_collapse./num_gms, 'xb', 'linewidth', 1.0)
hold on
plot(x_vals,p_collapse_mle_near, '-r', 'linewidth', 1.0)
title ('Fragility Curve', 'Interpreter','latex')
legh = legend('Observed collapse frequency', 'Maximum likelihood fit', 'location', 'southeast', 'Interpreter','latex');
set(legh)%, 'fontsize', 12)
hx = xlabel('Scale factor', 'Interpreter','latex');
hy = ylabel('Probability of collapse', 'Interpreter','latex');
axis([0 xmax 0 1])

disp(p_collapse_mle_near(x_vals == 1))
