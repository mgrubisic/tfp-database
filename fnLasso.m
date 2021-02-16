%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Isolation cost, LASSO regression

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	February 2021

% Description: 	Script performs LASSO regression on cost of isolation data
% set to assist in optimization and inverse design. This is the function
% version to be used with main script

% Open issues: 	(1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [costVec, intercept, lassoInfo] = fnLasso(XCost)
    %% prepare cost calculations
    isolDat     = readtable('../pastRuns/random200withTfb.csv');
    g           = 386.4;
    isolDat.beamWeight  = str2double(extractAfter(isolDat.beam,'X'));
    isolDat.roofWeight  = str2double(extractAfter(isolDat.roofBeam,'X'));
    isolDat.colWeight   = str2double(extractAfter(isolDat.col,'X'));

    % 624 ft of columns, 720 ft of beam, 360 ft of roof
    isolDat.bldgWeight  = 624*isolDat.colWeight + 720*isolDat.beamWeight + ...
        360*isolDat.roofWeight;

    % $1.25/lb of steel
    isolDat.steelCost   = 1.25*isolDat.bldgWeight;

    isolDat.kM          = (1/g)*(2*pi./isolDat.Tm).^2;
    % 2 layers of frame per direction
    Ws                  = 2227.5;
    W                   = 3037.5;
    isolDat.Vb          = (isolDat.moatGap.*isolDat.kM*2227.5)/2;
    isolDat.Vst         = (isolDat.Vb.*(Ws/W).^(1 - 2.5*isolDat.zetaM));
    isolDat.Vs          = isolDat.Vst./isolDat.RI;

    % Hogan: land cost is about 20% of the development cost ($1110/sf)
    landCostPerSqft     = 0.2*1110;
    isolDat.landCost    = landCostPerSqft/144*(90*12 + 2*isolDat.moatGap).^2;
    
    YCost               = isolDat.landCost + isolDat.steelCost;
    
    %% LASSO regression

    % Construct the lasso fit using 10-fold cross-validation
    % Use the largest Lambda value such that the mean squared error (MSE) 
    % is within one standard error of the minimum MSE.
    [B,lassoInfo]   = lasso(XCost,YCost,'CV',10);
    idxLambda1SE    = lassoInfo.Index1SE;
    costVec         = B(:,idxLambda1SE);
    intercept       = lassoInfo.Intercept(idxLambda1SE);
end
