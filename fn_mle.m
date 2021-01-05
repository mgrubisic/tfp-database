%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	MLE helper function

% Created by: 	Ya-Heng Yang
% 				University of California, Berkeley

% Date created:	November 2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [theta, beta] = fn_mle(IM, num_gms, num_collapse)
% Initial guess for the fragility function parameters theta and beta
% ** Use method of moments **
firstTry = [mean(log(IM)), std(log(IM))];

% Run optimization
options = optimset('MaxFunEvals',1000, 'GradObj', 'off'); %maximum 1000 iterations, gradient of the function not provided
x = fminsearch(@fit, firstTry, options, num_gms, num_collapse, IM) ;
theta = exp(x(1)); % return theta in linear space
beta = x(2);

% Objective function to be optimized
function [result] = fit(pa, num_gms, num_collapse, IM)
% estimated probabilities of collapse using the current fragility function parameter estimates
p = normcdf(log(IM), (pa(1)), pa(2));

% likelihood using the current fragility function parameter estimates
likelihood = binopdf(num_collapse', num_gms', p');

% no zero likelihood (because of log likelihood)
likelihood(likelihood == 0) = realmin;

% sum negative log likelihood (searching for a minimum value)
result = -sum(log(likelihood));
