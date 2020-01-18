featuresx = readmatrix("predx_for_classification.csv");
featuresy = readmatrix("predy_for_classification.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("label.csv");
labelsP   = allLabels(:, 4);

randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);


dataCount = 62906;
featureSubset = features(1:dataCount, :);
featureNames = [strcat("x", string(1:49)), strcat("y", string(1:49))];
labelSubset   = labels(1:dataCount);

stopPercen = 1
labelCountPercen = 0

%params: features, labels, featurenames, stop percentage, no. of
%thresholds, max depth

tree = makeDecisionTree(featureSubset, labelSubset, featureNames, stopPercen, 100, 10000000, labelCountPercen);


testFeatures = features(dataCount:end, :);
testLabels = labels(dataCount:end);

testAcc = simpleAccuracy(tree, testFeatures, testLabels);
testAcc = round(testAcc, 4)

nodeCount = treeNodeCount(tree)

trainAcc = simpleAccuracy(tree, featureSubset, labelSubset);
trainAcc = round(trainAcc, 4)
    
disp("-----------");


