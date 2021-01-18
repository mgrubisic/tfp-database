%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Marginal plot utility

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script plots marginals for individual params

% Open issues: 	(1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotMarginalSlices(constIdx, xIdx, fixIdx, x, y, ...
    hyp, meanfunc, covfunc ,inffunc, likfunc)

    [~,f]       = size(x);
    
    minX        = round(min(x),2);
%     maxX        = round(max(x),2);
%     minX        = round(prctile(x,5), 2);
    maxX        = round(prctile(x,95), 2);
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
    
    [a,b,~,~,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));
    
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
        evalMean = [a(idc)+sqrt(b(idc)) ; flip(a(idc)-sqrt(b(idc)), 1)];
        fill([pts; flip(pts,1)], evalMean, [7 7 7]/8)
        hold on;
        plot(pts, a(idc))
        
        % get points close to the desired v1's
        % tolerance: 5 steps of the constant variables
        % tolerance: 5 steps of the fix variables
        
        conditionIdxConst = (x(:,constIdx) < (constMed+5*stepX(constIdx)) ) & ...
            (x(:,constIdx) > (constMed-5*stepX(constIdx)));
        
        xPlot = x(conditionIdxConst,:);
        yPlot = y(conditionIdxConst,:);
        
        % this step is dodgy
        conditionIdxFix = (xPlot(:,fixIdx) < (vFix(i)+5*stepX(fixIdx)) ) & ...
            (xPlot(:,fixIdx) > (vFix(i)-5*stepX(fixIdx)) ) ;
        
        xPlot = xPlot(conditionIdxFix,:);
        yPlot = yPlot(conditionIdxFix,:);
        
        % plot data
        collapsedIdx   = yPlot == 1;
        notIdx = yPlot == -1;
        
        hold on;
        plot(xPlot(collapsedIdx, xIdx), yPlot(collapsedIdx), 'x', 'MarkerEdgeColor', [0.8500 0.3250 0.0980], 'MarkerSize', 10); 
        plot(xPlot(notIdx, xIdx), yPlot(notIdx), '+', 'MarkerEdgeColor', [0 0.4470 0.7410], 'MarkerSize', 10); 
        
        % put vertical line where 5% collapse is (= collapse eval of -0.90)
        [~, idx] = min(abs(a(idc) + 0.90));
        xl = xline(pts(idx), '-.k', {'5\%', 'collapse'}, 'fontsize', 20);
        xl.Interpreter = 'latex';
        
        xlabel('Gap ratio', 'fontsize', 14, 'Interpreter','latex')
        %ylabel(['$T_2$ ratio = ', num2str(vFix(i),3)], 'Interpreter','latex')
        ylabel('Collapse prediction', 'fontsize', 14, 'Interpreter','latex')
        legend('Standard deviation', 'Collapse prediction', 'Collapse', 'No collapse', 'fontsize', 18, 'Interpreter','latex')
        set(gca,'FontSize', 18)
        yticks([-1 0 1])
        xlim([minX(1) maxX(1)])
        
    end
    sgtitle('Marginal collapse predictions across $T_M$ ratio values', 'Interpreter', 'LaTeX')
end