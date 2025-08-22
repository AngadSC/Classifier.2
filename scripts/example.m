
[FN400_values, LPP_values] = feature_extraction(data,test);
[OldNew_labels, HitsMisses_labels, CR_FA_labels, Perceived_labels] = label_extraction(test);
[ROC_data, AUC_results] = Univariate_Classifier(FN400_values, LPP_values, OldNew_labels, HitsMisses_labels, CR_FA_labels, Perceived_labels);
