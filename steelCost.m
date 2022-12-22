%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             	Steel cost regression

% Created by: 	Huy Pham
% 				University of California, Berkeley

% Date created:	February 2021

% Description: 	Script performs linear regression on steel cost as a
% function of base shear, to be used for grid search cost calcs

% Open issues: 	(1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function steelCoef = steelCost(isolDat)
    %% Grab data
    g           = 386.4;

    % scaling Sa(Tm) for damping, ASCE Ch. 17
    zetaRef     = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];
    BmRef       = [0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0];
    isolDat.Bm  = interp1(zetaRef, BmRef, isolDat.zetaM);

    %% Calculate steel cost of existing data
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
    isolDat.Dm          = g*1.017.*isolDat.Tm./(4*pi^2.*isolDat.Bm);
    isolDat.Vb          = (isolDat.Dm.*isolDat.kM*2227.5)/2;
    isolDat.Vst         = (isolDat.Vb.*(Ws/W).^(1 - 2.5*isolDat.zetaM));
    isolDat.Vs          = isolDat.Vst./isolDat.RI;
    
    %% Use simple least squares regression
    linMdl      = fitlm(isolDat.Vs, isolDat.steelCost);
    steelCoef   = linMdl.Coefficients.Estimate;
end
