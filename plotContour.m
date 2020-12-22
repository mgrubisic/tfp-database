%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Contour plot utility

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	November 2020

% Description: 	Script plots a contour for GP

% Open issues: 	(1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotContour(constIdx, xIdx, yIdx, x, y, ...
    hyp, meanfunc, covfunc ,inffunc, likfunc)
    

    [~,f]       = size(x);
    
    minX        = round(min(x),2);
    maxX        = round(max(x),2);
    midX        = round(median(x),2);
    stepX       = (maxX-minX)/50;

    vConst =  [minX(constIdx) midX(constIdx) maxX(constIdx)];
    
    [tempX, tempY] = meshgrid(minX(xIdx):stepX(xIdx):maxX(xIdx), ...
            minX(yIdx):stepX(yIdx):maxX(yIdx));
        
    tmp = [tempX(:) tempY(:)]; n = length(tmp);
        
    figure
    
    for i = 1:length(vConst)
        
        subplot(1,3,i)
        
        t = zeros(n,f);
        
        %creates points at which the model will be evaluated
        for j = 1:f
            if j == constIdx
                t(:,j) = ones(n,1)*vConst(i);
            elseif j == xIdx
                t(:,j) = tmp(:,1);
            elseif j == yIdx
                t(:,j) = tmp(:,2);
            else
                t(:,j) = ones(n,1)*midX(j);
            end
        end
        
        %a is expected value, b is sd of expected value, lp is the probabilities used to print the contour curves
        [a,b,c,d,lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1)); 
        
        % get points close to the desired vConst's
        xPlot = x(( (x(:,constIdx) < vConst(i)+5*stepX(constIdx)) & ...
            (x(:,constIdx) > vConst(i)-5*stepX(constIdx)) ),:);
        yPlot = y(( (x(:,constIdx) < vConst(i)+5*stepX(constIdx)) & ...
            (x(:,constIdx) > vConst(i)-5*stepX(constIdx)) ),:);
        
        % plot data
        collapsedIdx   = yPlot == 1;
        notIdx = yPlot == -1;
        
        hold on;
        plot(xPlot(collapsedIdx,xIdx), xPlot(collapsedIdx,yIdx), 'r+'); 
        plot(xPlot(notIdx,xIdx), xPlot(notIdx,yIdx), 'b+');
        
        xlabel('Gap ratio','Interpreter','latex')
        ylabel('$T_2$ ratio','Interpreter','latex')
        contour(tempX, tempY, reshape(exp(lp), size(tempX)), [0.0:0.1:1.0]);
        
        
    end
    colorbar
    %sgtitle('Probability of collapse at $\zeta_M = 0.10, 0.15, 0.20$', 'Interpreter', 'LaTeX')

end