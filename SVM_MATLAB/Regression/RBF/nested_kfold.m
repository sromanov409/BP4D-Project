featuresx = readmatrix("../data/predx_for_regression.csv");
featuresy = readmatrix("../data/predy_for_regression.csv");
featuresP = [featuresx, featuresy];
allLabels = readmatrix("../data/angle.csv");
labelsP   = allLabels(:, 3);

randIdxs = randperm(length(labelsP));
labels = labelsP(randIdxs);
features = featuresP(randIdxs, :);

labels = translateYaw(labels);

% PARAMS
usePercent     = 0.05;
outerFoldCount = 5;
innerFoldCount = 3;
parameterCount = 3;
param1 = [20 50 100]; % KernelScale
param2 = [100 150 200]; % C
param3 = [0.3 0.5 0.7]; % Epsilon
% END

hyperparameters = cartProd(param1, param2, param3);

cutCount = round(usePercent * length(labels));
labels = labels(1:cutCount);
features = features(1:cutCount, :);

foldErrors = zeros(outerFoldCount, 1);
foldParameters = cell(outerFoldCount, parameterCount);


for i = 1:outerFoldCount
    [outer_train_features, outer_train_labels, outer_test_features, outer_test_labels] ...
        = splitFolds(i, outerFoldCount, features, labels);
    
    disp("Outer fold " + i);
    
    [error, params] = innerfolds(outer_train_features, outer_train_labels, ...
    outer_test_features, outer_test_labels, innerFoldCount, hyperparameters);
    
    disp("Outer error: " + error);
    disp("Outer parameters: " + params{1} + ", " + params{2} + ", " + params{3});
    foldErrors(i) = error;
    foldParameters(i, :) = params;
end

meanError = mean(foldErrors)
















