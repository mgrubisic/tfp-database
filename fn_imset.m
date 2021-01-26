%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Weighted IMSE evaluation

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	January 2021

% Description: 	Script evaluates the integrated mean squared error of a
% dataset and new points, to be used with optimizer

% Open issues: 	(1) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [IMSEt] = fn_imset(xnew, hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk, T)
    % if ys2 is unaffected by y observations, create dummy one w/ k+1 rows
    yDummy      = ones(length(xk) + 1, 1);
    xSet        = [xk; xnew];
    [~,s2k,~,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, xSet, yDummy,...
        xk);
    
    [ymu,ys2,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk,...
        xk);
    
    % IMSEt (Picheny)
    PDFx        = exp(lp);

    sigE        = 0.05*(max(yk)- min(yk));
    Wx          = 1./sqrt(2*pi*(sigE^2 + ys2)) .* ...
        exp((-1/2)*((ymu - T).^2./sigE^2 + ys2.^2));

    IMSEt       = sum(s2k.*Wx.*PDFx);
end