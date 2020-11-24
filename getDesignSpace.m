%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Design space finder utility

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script finds the design space given desired probability of
% failure and variables of interest

% Open issues: 	(1) Currently limited to finding space over two variables

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [designSpace, boundLine] = getDesignSpace(varX, varY, probDesired, probTol, x, y, hyp, meanfunc, covfunc, inffunc, likfunc)
    [~,f]       = size(x);
    
    minX        = round(min(x),1);
    maxX        = round(max(x),1);
    midX        = round(median(x),2);
    stepX       = (maxX-minX)/50;

    [tempX, tempY] = meshgrid(minX(varX):stepX(varX):maxX(varX), ...
                minX(varY):stepX(varY):maxX(varY));

    tmp = [tempX(:) tempY(:)]; n = length(tmp);

    t = zeros(n,f);

    %creates points at which the model will be evaluated
    for j = 1:f
        if j == varX
            t(:,j) = tmp(:,1);
        elseif j == varY
            t(:,j) = tmp(:,2);
        else
            t(:,j) = ones(n,1)*midX(j);
        end
    end

    [~,~,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ...
        x, y, t, ones(n, 1));

    minSpace    = [tempX(:) tempY(:) exp(lp)];
    
    designSpace = minSpace((minSpace(:, end) <= probDesired),:);
    boundLine   = minSpace((minSpace(:, end) <= probDesired + probTol) ...
        & (minSpace(:, end) >= probDesired - probTol), :);
    
    % get points close to the desired vConst's
    for j = 1:f
        if (j ~= varX) || (j ~= varY)
            xPlot = x(( (x(:,j) < midX(j) + 5*stepX(j)) & ...
                (x(:,j) > midX(j) - 5*stepX(j)) ),:);
            yPlot = y(( (x(:,j) < midX(j) + 5*stepX(j)) & ...
                (x(:,j) > midX(j) - 5*stepX(j)) ),:);
        end
    end
    
    figure
    % plot data
    collapsedIdx   = yPlot == 1;
    notIdx = yPlot == -1;
    hold on;
    plot(xPlot(collapsedIdx,varX), xPlot(collapsedIdx,varY), 'r+'); 
    plot(xPlot(notIdx,varX), xPlot(notIdx,varY), 'b+');
    plot(boundLine(:,1), boundLine(:,2))
end
