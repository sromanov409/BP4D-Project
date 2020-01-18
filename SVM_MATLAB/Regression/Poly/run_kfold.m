featuresx = readmatrix("../data/predx_for_regression.csv");
featuresy = readmatrix("../data/predy_for_regression.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("../data/angle.csv");
labelsP   = allLabels(:, 3);

randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);

labels = translateYaw(labels);

[features, labels] = removeOutliers(features, labels);

% PARAMS
usePercent     = 0.1;
params = {2, 2, 0.048};
% END

cutCount = round(usePercent * length(labels));
labels = labels(1:cutCount);
features = features(1:cutCount, :);

output = kfold_function(10, features, labels, 'polynomial', params);