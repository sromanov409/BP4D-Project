featuresx = readmatrix("../data/predx_for_classification.csv");
featuresy = readmatrix("../data/predy_for_classification.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("../data/label.csv");
labelsP   = allLabels(:, 2);

randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);


% PARAMS
usePercent     = 0.2;
outerFoldCount = 5;
innerFoldCount = 3;
parameterCount = 2;
param1 = [3 4] % Order
param2 = [3 6 10]; % C
% END 

hyperparameters = cartProd(param1, param2);

cutCount = round(usePercent * length(labels));
labels = labels(1:cutCount);
features = features(1:cutCount, :);

foldAccuracies = zeros(outerFoldCount, 1);
foldParameters = cell(outerFoldCount, parameterCount);


for i = 1:outerFoldCount
    [outer_train_features, outer_train_labels, outer_test_features, outer_test_labels] ...
        = splitFolds(i, outerFoldCount, features, labels);
    
    disp("Outer fold " + i);
    
    [accuracy, params] = innerfolds(outer_train_features, outer_train_labels, ...
    outer_test_features, outer_test_labels, innerFoldCount, hyperparameters);
    
    disp("Outer accuracy: " + accuracy);
    disp("Outer parameters: " + params{1} + ", " + params{2});
    foldAccuracies(i) = accuracy;
    foldParameters(i, :) = params;
end

meanAccuracy = mean(foldAccuracies)
















