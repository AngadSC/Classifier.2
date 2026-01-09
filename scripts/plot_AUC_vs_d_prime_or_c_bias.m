% Load LDA/SVM results
transformation_type = 'raw';  % Change as needed
AUC_LDA=load(sprintf('/Users/pawluk/Desktop/classifiers/scripts_faisal/multivariate/LDA/LDA_results_%s.mat', transformation_type));
AUC_SVM=load(sprintf('/Users/pawluk/Desktop/classifiers/scripts_faisal/multivariate/SVM/SVM_results_%s.mat', transformation_type));

% Extract participant IDs
subjects = fieldnames(AUC_LDA.AUC_all);

% Classification tasks and labels
tasks = {'OldNew', 'HitMiss', 'FAvsCR'};
task_labels = {'Old/New', 'Hit/Miss', 'FA/CR'};

% Loop through each classification task
for s = 1:length(subjects)
    AUC_LDA_OldNew(s) = AUC_LDA.AUC_all.(subjects{s}).(tasks{1});
    AUC_LDA_HitMiss(s) = AUC_LDA.AUC_all.(subjects{s}).(tasks{2});
    AUC_LDA_FAvsCR(s) = AUC_LDA.AUC_all.(subjects{s}).(tasks{3});
    AUC_SVM_OldNew(s) = AUC_SVM.AUC_all.(subjects{s}).(tasks{1});
    AUC_SVM_HitMiss(s) = AUC_SVM.AUC_all.(subjects{s}).(tasks{2});
    AUC_SVM_FAvsCR(s) = AUC_SVM.AUC_all.(subjects{s}).(tasks{3});
end
AUC_LDA_FAvsCR(isnan(AUC_LDA_FAvsCR))=[]; AUC_SVM_FAvsCR(isnan(AUC_SVM_FAvsCR))=[];
AUC_LDA_OldNew = AUC_LDA_OldNew(:); AUC_LDA_HitMiss = AUC_LDA_HitMiss(:); AUC_LDA_FAvsCR = AUC_LDA_FAvsCR(:);
AUC_SVM_OldNew = AUC_SVM_OldNew(:); AUC_SVM_HitMiss = AUC_SVM_HitMiss(:); AUC_SVM_FAvsCR = AUC_SVM_FAvsCR(:);
load('dprime.mat'); load('c_bias.mat'); load('abs_c_bias.mat')

% Pearson correlations (r and p-value), also slopes and intercepts
[R, P] = corrcoef(AUC_LDA_OldNew,d_prime);
Pearson.LDA.OldNew = [sprintf('r = %+.4f',R(1,2)); sprintf('p = % .4f',P(1,2))];
r_slope.LDA.OldNew = R(1,2) .* (std(AUC_LDA_OldNew)./std(d_prime));
r_intercept.LDA.OldNew = mean(AUC_LDA_OldNew) - (r_slope.LDA.OldNew.*mean(d_prime));
[R, P] = corrcoef(AUC_LDA_HitMiss,d_prime);
Pearson.LDA.HitMiss = [sprintf('r = %+.4f',R(1,2)); sprintf('p = % .4f',P(1,2))];
r_slope.LDA.HitMiss = R(1,2) .* (std(AUC_LDA_HitMiss)./std(d_prime));
r_intercept.LDA.HitMiss = mean(AUC_LDA_HitMiss) - (r_slope.LDA.HitMiss.*mean(d_prime));
[R, P] = corrcoef(AUC_SVM_OldNew,d_prime);
Pearson.SVM.OldNew = [sprintf('r = %+.4f',R(1,2)); sprintf('p = % .4f',P(1,2))];
r_slope.SVM.OldNew = R(1,2) .* (std(AUC_SVM_OldNew)./std(d_prime));
r_intercept.SVM.OldNew = mean(AUC_SVM_OldNew) - (r_slope.SVM.OldNew.*mean(d_prime));
[R, P] = corrcoef(AUC_SVM_HitMiss,d_prime);
Pearson.SVM.HitMiss = [sprintf('r = %+.4f',R(1,2)); sprintf('p = % .4f',P(1,2))];
r_slope.SVM.HitMiss = R(1,2) .* (std(AUC_SVM_HitMiss)./std(d_prime));
r_intercept.SVM.HitMiss = mean(AUC_SVM_HitMiss) - (r_slope.SVM.HitMiss.*mean(d_prime));
% FAvsCR removing NaN
d_prime_FA_CR=d_prime; d_prime_FA_CR([42,44])=[];
[R, P] = corrcoef(AUC_LDA_FAvsCR,d_prime_FA_CR);
Pearson.LDA.FAvsCR = [sprintf('r = %+.4f',R(1,2)); sprintf('p = % .4f',P(1,2))];
r_slope.LDA.FAvsCR = R(1,2) .* (std(AUC_LDA_FAvsCR)./std(d_prime_FA_CR));
r_intercept.LDA.FAvsCR = mean(AUC_LDA_FAvsCR) - (r_slope.LDA.FAvsCR.*mean(d_prime));
[R, P] = corrcoef(AUC_SVM_FAvsCR,d_prime_FA_CR);
Pearson.SVM.FAvsCR = [sprintf('r = %+.4f',R(1,2)); sprintf('p = % .4f',P(1,2))];
r_slope.SVM.FAvsCR = R(1,2) .* (std(AUC_SVM_FAvsCR)./std(d_prime_FA_CR));
r_intercept.SVM.FAvsCR = mean(AUC_SVM_FAvsCR) - (r_slope.SVM.FAvsCR.*mean(d_prime));

% AUC vs d' plots
figure;
subplot(2,3,1);
%plot(d_prime,AUC_LDA_OldNew,'o');
plot(d_prime,AUC_LDA_OldNew,'o',[0,4],[r_intercept.LDA.OldNew, r_slope.LDA.OldNew*4+r_intercept.LDA.OldNew],'r');
ylim([0.2,0.8]);
xlabel('d'''); ylabel('AUC'); title('Old/New (LDA)');
subplot(2,3,2);
%plot(d_prime,AUC_LDA_HitMiss,'o');
plot(d_prime,AUC_LDA_HitMiss,'o',[0,4],[r_intercept.LDA.HitMiss, r_slope.LDA.HitMiss*4+r_intercept.LDA.HitMiss],'r');
ylim([0.2,0.8]);
xlabel('d'''); ylabel('AUC'); title('Hit/Miss (LDA)');
subplot(2,3,3)
%plot(d_prime_FA_CR,AUC_LDA_FAvsCR,'o');
plot(d_prime_FA_CR,AUC_LDA_FAvsCR,'o',[0,4],[r_intercept.LDA.FAvsCR, r_slope.LDA.FAvsCR*4+r_intercept.LDA.FAvsCR],'r');
ylim([0.2,0.8]);
xlabel('d'''); ylabel('AUC'); title('FA/CR (LDA)');
subplot(2,3,4);
%plot(d_prime,AUC_SVM_OldNew,'o');
plot(d_prime,AUC_SVM_OldNew,'o',[0,4],[r_intercept.SVM.OldNew, r_slope.SVM.OldNew*4+r_intercept.SVM.OldNew],'r');
ylim([0.2,0.8]);
xlabel('d'''); ylabel('AUC'); title('Old/New (SVM)');
subplot(2,3,5);
%plot(d_prime,AUC_SVM_HitMiss,'o');
plot(d_prime,AUC_SVM_HitMiss,'o',[0,4],[r_intercept.SVM.HitMiss, r_slope.SVM.HitMiss*4+r_intercept.SVM.HitMiss],'r');
ylim([0.2,0.8]);
xlabel('d'''); ylabel('AUC'); title('Hit/Miss (SVM)');
subplot(2,3,6)
%plot(d_prime_FA_CR,AUC_SVM_FAvsCR,'o');
plot(d_prime_FA_CR,AUC_SVM_FAvsCR,'o',[0,4],[r_intercept.SVM.FAvsCR, r_slope.SVM.FAvsCR*4+r_intercept.SVM.FAvsCR],'r');
ylim([0.2,0.8]);
xlabel('d'''); ylabel('AUC'); title('FA/CR (SVM)');
%{
% AUC vs c plots
figure;
subplot(1,3,1);
plot(c_bias,AUC_OldNew,'o');
xlabel('c'); ylabel('AUC'); title('Old/New (LDA)');
subplot(1,3,2);
plot(c_bias,AUC_HitMiss,'o');
xlabel('c'); ylabel('AUC'); title('Hit/Miss (LDA)');
subplot(1,3,3);
plot(c_bias,AUC_FAvsCR,'o');
xlabel('c'); ylabel('AUC'); title('FA/CR (LDA)');

% AUC vs |c| plots
figure;
subplot(1,3,1);
plot(abs_c_bias,AUC_OldNew,'o');
xlabel('|c|'); ylabel('AUC'); title('Old/New (LDA)');
subplot(1,3,2);
plot(abs_c_bias,AUC_HitMiss,'o');
xlabel('|c|'); ylabel('AUC'); title('Hit/Miss (LDA)');
subplot(1,3,3);
plot(abs_c_bias,AUC_FAvsCR,'o');
xlabel('|c|'); ylabel('AUC'); title('FA/CR (LDA)');
%}