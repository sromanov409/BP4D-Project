featuresx = readmatrix("predx_for_regression.csv");
featuresy = readmatrix("predy_for_regression.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("angle.csv");
labelsP   = allLabels(:, 1);

randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);

% PARAMS
usePercent        = 0.1;
trainSplitPercent = 0.8;
epsilon           = 1.5;
% END

cutCount = usePercent * length(labels);
labels = labels(1:cutCount);
features = features(1:cutCount, :);

trainDataCount= round(cutCount * trainSplitPercent);
featureSubset = features(1:trainDataCount, :);
featureNames  = [strcat("x", string(1:49)), strcat("y", string(1:49))];
labelSubset   = labels(1:trainDataCount);

Mdl = fitrsvm(featureSubset,labelSubset, 'KernelFunction','linear', 'Epsilon',epsilon);

testFeatures  = features(trainDataCount:end, :);
testLabels    = labels(trainDataCount:end);

testModelRegression(Mdl, testFeatures, testLabels)