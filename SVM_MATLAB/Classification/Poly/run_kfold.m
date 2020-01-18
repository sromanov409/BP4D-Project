featuresx = readmatrix("../data/predx_for_classification.csv");
featuresy = readmatrix("../data/predy_for_classification.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("../data/label.csv");
labelsP   = allLabels(:, 3);

randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);

% PARAMS
usePercent = 0.2;
params = {2, 1.4};
% END

cutCount = round(usePercent * length(labels));
labels = labels(1:cutCount);
features = features(1:cutCount, :);

output = kfold_function(10, features, labels, 'polynomial', params);