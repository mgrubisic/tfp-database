%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Marginal plot utility

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script plots marginals for individual params

% Open issues: 	(1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotMarginalSlices(constIdx, xIdx, fixIdx, x, y, hyp, meanfunc, covfunc ,inffunc, likfunc)

    [~,f]       = size(x);
    
    minX        = round(min(x),1);
    maxX        = round(max(x),1);
    midX        = round(median(x),2);
    stepX       = (maxX-minX)/50;
    
    constMed =  midX(constIdx);
    
    [tempX, tempY] = meshgrid(minX(xIdx):stepX(xIdx):maxX(xIdx), ...
            minX(fixIdx):stepX(fixIdx):maxX(fixIdx));
        
    tmp = [tempX(:) tempY(:)]; n = length(tmp);
    
%     t = [t1(:) t2(:)]; n = length(t);
%     t = [ones(n,1)*constMed t ones(n,1)*midX(4:f)];

    t = zeros(n,f);
    
    %creates points at which the model will be evaluated
        for j = 1:f
            if j == constIdx
                t(:,j) = ones(n,1)*constMed;
            elseif j == xIdx
                t(:,j) = tmp(:,1);
            elseif j == fixIdx
                t(:,j) = tmp(:,2);
            else
                t(:,j) = ones(n,1)*midX(j);
            end
        end
    
    [a,b,~,~,~] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
    
%     v3  = [midX(3)-10*stepX(3) midX(3) midX(3)+10*stepX(3)];

    vFix  = [midX(fixIdx)-10*stepX(fixIdx) ...
        midX(fixIdx) ...
        midX(fixIdx)+10*stepX(fixIdx)];
    
    hold off
    figure
    
    for i = 1:length(vFix)
        subplot(length(vFix),1,i)
        
        %     num = round((v3(i)-minX(3))/stepX(3));
        %     q = gl*num + 1;
        %     v = q + gl - 1 ;
        %     pts = t(q:v,2); %y values are changing
        
        % find where is the interval of fixed variable
        [~, ix] = min(abs(t(:,fixIdx) - vFix(i)));
        
        % find cycles of the iterated x variable
        % grab index corresponding to the median fix variable
        [pts, idc] = unique(t(:,xIdx));
        idc = idc + ix;
        
        % plot the marginal with stdev intervals
        evalMean = [a(idc)+2*sqrt(b(idc)) ; flip(a(idc)-2*sqrt(b(idc)), 1)];
        fill([pts; flip(pts,1)], evalMean, [7 7 7]/8)
        hold on;
        plot(pts, a(idc))
        
        % get points close to the desired v1's
        % tolerance: 2 steps of the constant variables
        % tolerance: 2 steps of the fix variables
        
        xPlot = x(( (x(:,constIdx) < constMed+2*stepX(constIdx)) & ...
            (x(:,constIdx) > constMed-2*stepX(constIdx)) ),:);
        yPlot = y(( (x(:,constIdx) < constMed+2*stepX(constIdx)) & ...
            (x(:,constIdx) > constMed-2*stepX(constIdx)) ),:);
        
        xPlot = xPlot(( (xPlot(:,fixIdx) < vFix(i)+5*stepX(fixIdx)) & ...
            (xPlot(:,fixIdx) > vFix(i)-5*stepX(fixIdx)) ),:);
        yPlot = yPlot(( (xPlot(:,fixIdx) < vFix(i)+5*stepX(fixIdx)) & ...
            (xPlot(:,fixIdx) > vFix(i)-5*stepX(fixIdx)) ),:);
        
        % plot data
        collapsedIdx   = yPlot == 1;
        notIdx = yPlot == -1;
        
        hold on;
        plot(xPlot(collapsedIdx, xIdx), yPlot(collapsedIdx), 'r+'); 
        plot(xPlot(notIdx, xIdx), yPlot(notIdx), 'b+');
        
        xlabel('x variable','Interpreter','latex')
        ylabel('collapsed?','Interpreter','latex')
        
    end
    
    sgtitle('marginals for 3 sets of median values', 'Interpreter', 'LaTeX')
end