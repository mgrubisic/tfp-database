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
    k           = length(xk);
    yDummy      = ones(k + 1, 1);
    xSet        = [xk; xnew];
    
    % from Schur's complement formula
    Ck          = covfunc(hyp.cov, xk);
    CkNewTemp   = covfunc(hyp.cov, xSet);
    cnew        = CkNewTemp(1:(end-1),end);
    sigVar      = Ck(1,1);
    CkNew       = [sigVar cnew'; cnew Ck];
    CkNewInv    = [1 zeros(k,1)'; -inv(Ck)*cnew eye(k)]*...
        [1/(sigVar - cnew'*inv(Ck)*cnew) zeros(k,1)'; zeros(k,1) inv(Ck)]* ...
        [1 -cnew'*inv(Ck); zeros(k,1) eye(k)];
    
%     % this is probably inefficient to optimize
%     % see Picheny paper on calculating s2k with Schur's complement formula
%     [~,s2k,~,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, xSet, yDummy,...
%         xk);
%     
%     [ymu,ys2,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, xk, yk,...
%         xk);
%     
%     % IMSEt (Picheny)
%     PDFx        = exp(lp);
% 
%     sigE        = 0.05*(max(yk)- min(yk));
%     Wx          = 1./sqrt(2*pi*(sigE^2 + ys2)) .* ...
%         exp((-1/2)*((ymu - T).^2./sigE^2 + ys2.^2));
% 
%     IMSEt       = sum(s2k.*Wx.*PDFx);

    
end