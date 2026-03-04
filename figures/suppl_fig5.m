% Define positions in V1 in mm (distance from foveal representation)
dx  = 0.5; % mm
pos = 0:dx:50; 
 
 
% Center and size of V1 pRFs
A = 17.3; B = 0.75; % horton and hoyt, 1991
Ecc0 = B*(exp(pos/A) -1);  % deg
Sz   = Ecc0 * .06+.3; % where from? Dumoulin and Wandell, figure 9
ECC = linspace(0, max(Ecc0)*1.2, 500)';
PRF = exp(-1/2*   ((ECC-Ecc0)./(Sz)).^2);
PRF = PRF ./ max(PRF);

% Center and size of V1 pSFs
m = 0.15; b = 0.17;
Per0 = Ecc0*m+b; % Ha et al. slope and sigma parameters
BW   = 2.2;%2.2; % in σ
PER = linspace(0,max(Per0)*2, 500)';
PSF = exp(-1/2 * ((log2(PER)-log2(Per0))/BW).^2);
PSF = PSF ./ max(PSF);

% kernel for smoothing pRF and pSF in V1
n = .5; % power law nonlinearity after pooling

kernel_sigma = 3.2; % in mm, Population point image size from Dumoulin and Harvey 2011 (Table 1)
V2kernel = exp(-1/2*((-20:dx:20)/kernel_sigma).^2); % gausswin(64)';
V2kernel = V2kernel / sum(V2kernel);

PRF2 = conv2(PRF, V2kernel, 'same');
PRF2 = PRF2 ./ max(PRF2);
PRF2NL = PRF2.^n; 

PSF2 = conv2(PSF, V2kernel, 'same');
PSF2 = PSF2 ./ max(PSF2);
PSF2NL = PSF2.^n;


% Create a new figure for the plots
figure(1), tiledlayout(2,3,"TileSpacing","compact");

% Set figure size in inches, keeping aspect ratio ~1.33
set(gcf,'Units','inches','Position',[1 1 8 4]); 

% choose a location to plot halfway along V1
k = (length(Ecc0)+1)/2;

%% PRFs
% V1 pRFs 
showPRFs(pos, ECC,  PRF, Ecc0, 'V1 PRFs', 'Eccentricity (deg)', pos(k), 'r')
showPRFs(pos, ECC,  PRF2, Ecc0, 'V1 Pooled', 'Eccentricity (deg)', pos(k), 'g')

% V1 and V2 pRF example
nexttile();
plot(ECC, PRF(:,k), 'r', ECC, PRF2(:,k),'g', LineWidth=2); hold on;
plot(ECC, PRF2NL(:,k), 'b--', LineWidth=1)
xlabel('Eccentricity'); ylabel('Response'); legend('V1', 'Pooled V1', 'Normalized V1', 'Location','eastoutside')
xlim(4*[-Sz(k) +Sz(k)] + Ecc0(k))
set(gca, FontSize=13)


%% PSFs

% V1 pSFs
showPRFs(pos, PER,  PSF, Per0,'V1 PSFs', 'Period (deg)', pos(k), 'r')
showPRFs(pos, PER,  PSF2, Per0,'V1 Pooled', 'Period (deg)', pos(k), 'g')

% V1 and V2 pSF example
nexttile(); 
plot(PER, PSF(:,k), 'r', PER, PSF2(:,k),'g--', LineWidth=2); hold on
plot( PER, PSF2NL(:,k), 'b--', LineWidth=1)

set(gca, 'XScale', 'log')
xlabel('Period (deg)'); ylabel('Response'); %legend('V1', 'Pooled V1', 'Normalized V1', 'Location','eastoutside')
set(gca, FontSize=13)


%% Check the sizes of PRFs/PSFs

PRF1  = fit(ECC, PRF(:,k), 'gauss1');
PRF2 = fit(ECC, PRF2(:,k), 'gauss1');
PRF3 = fit(ECC, PRF2NL(:,k), 'gauss1');

PSF1 = fit(log2(ECC(2:end)), PSF(2:end,k), 'gauss1');
PSF2 = fit(log2(ECC(2:end)), PSF2(2:end,k), 'gauss1');
PSF3 = fit(log2(ECC(2:end)), PSF2NL(2:end,k), 'gauss1');

V1          = [PRF1.c1; PSF1.c1]/sqrt(2);
Pooled      = [PRF2.c1; PSF2.c1]/sqrt(2);
Normalized  = [PRF3.c1; PSF3.c1]/sqrt(2);

T = table(V1, Pooled, Normalized, 'RowNames',{'PRF Size' 'PSF Size'});

% **********
function showPRFs(pos, ECC,  PRF, Ecc0, title_str, ystr, xl, color)
nexttile();
imagesc(pos, ECC,  PRF); axis xy; hold on
plot(pos, Ecc0, 'k')
xlabel('Position (mm)'); 
ylabel(ystr)

xlim([0 45])
title(title_str)
set(gca, 'FontSize', 13)
if exist('xl', 'var')    
    xline(xl, Color=color, LineWidth=1)    
end

end