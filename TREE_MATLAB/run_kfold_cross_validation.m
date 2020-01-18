ACTION_UNIT = 5; % CHANGE THIS TO YOUR ACTION UNIT NUMBER.
NUMBER_OF_FOLDS = 10; % This should be 10 for your final results.

BETA = 1; % Leave this at 1.
featuresx = readmatrix("predx_for_classification.csv");
featuresy = readmatrix("predy_for_classification.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("label.csv");
labelsP   = allLabels(:, ACTION_UNIT);

% Shuffling of the data.
randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);

featureNames = [strcat("x", string(1:49)), strcat("y", string(1:49))];

[accuracy, recall, precision, fmeasure, output] = kfold_function(NUMBER_OF_FOLDS, BETA, features, labels, featureNames);