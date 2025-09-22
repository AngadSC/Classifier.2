
resultsDir = '/Users/faisalanqouor/Desktop/Research/classification_Matt_Results/cleaned';
resultsFile = fullfile(resultsDir, 'Correlation_dprime_AUC.mat');

if exist(resultsFile, 'file')
    load(resultsFile, 'correlation_dprime_auc');
else
    error('not found');
end

correlation_results.CR_FA.FN400_P_Value

p_FN400 = correlation_dprime_auc.CR_FA.FN400.P_Value;
p_LPP = correlation_dprime_auc.CR_FA.LPP.P_Value;


figure;
plot(p_FN400, p_LPP, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k', 'MarkerSize', 8);
xlabel('FN400 p-value');
ylabel('LPP p-value');
title('FA/CR p-values: FN400 vs LPP');
grid on;
xlim([0 1]);
ylim([0 1]);


hold on;
yline(0.05, '--r', 'p = 0.05');
xline(0.05, '--r', 'p = 0.05');
hold off;

=
saveas(gcf, fullfile(resultsDir, 'FA_CR_pvalues_plot.png'));

disp('Plot saved successfully.');
