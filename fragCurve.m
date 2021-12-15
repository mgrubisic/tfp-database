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
% conference paper:
% num_collapse = [2, 25, 43]; 

% higher res
num_collapse = [0, 2, 2];
num_gms = [59, 50, 44];

% estimate fragility function using MLE method
[theta_hat_mle, beta_hat_mle] = fn_mle(IM, num_gms, num_collapse);

% compute fragility functions using estimated parameters
xmax = 4;
x_vals = 0.01:0.01:xmax;
p_collapse_mle_near = normcdf((log(x_vals/theta_hat_mle))/beta_hat_mle); % compute fragility function using equation 1 and estimated parameters

%% plot resulting fragility functions
figure
plot(IM,num_collapse./num_gms, 'xb', 'linewidth', 1.0, 'MarkerSize', 14)
hold on
plot(x_vals,p_collapse_mle_near, '-r', 'linewidth', 1.0)
yl = yline(0.05, '-.k', {'5\% collapse'}, 'fontsize', 18, 'linewidth', 1.0);
yl.Interpreter = 'latex';
legh = legend('Observed collapse frequency', 'Maximum likelihood fit', 'location', 'best', 'Interpreter','latex');
set(legh, 'fontsize', 18);
set(gca,'FontSize', 18)
hx = xlabel('Scale factor', 'fontsize', 18, 'Interpreter','latex');
hy = ylabel('Probability of collapse', 'fontsize', 18, 'Interpreter','latex');
axis([0 xmax 0 1])

disp(p_collapse_mle_near(x_vals == 1))
