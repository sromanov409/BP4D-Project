featuresx = readmatrix("predx_for_classification.csv");
featuresy = readmatrix("predy_for_classification.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("label.csv");
labelsP   = allLabels(:, 1);

randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);

% PARAMS
usePercent        = 0.01;
trainSplitPercent = 0.8;
% END

cutCount = usePercent * length(labels);
labels = labels(1:cutCount);
features = features(1:cutCount, :);

trainDataCount= round(cutCount * trainSplitPercent);
featureSubset = features(1:trainDataCount, :);
featureNames  = [strcat("x", string(1:49)), strcat("y", string(1:49))];
labelSubset   = labels(1:trainDataCount);

Mdl = fitcsvm(featureSubset,labelSubset, 'KernelFunction','linear', 'BoxConstraint',1);

testFeatures  = features(trainDataCount:end, :);
testLabels    = labels(trainDataCount:end);

testModelClassification(Mdl, testFeatures, testLabels)