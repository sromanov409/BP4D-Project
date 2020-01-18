featuresx = readmatrix("../data/predx_for_classification.csv");
featuresy = readmatrix("../data/predy_for_classification.csv");
featuresP = [featuresx, featuresy];
labelsP = readmatrix("../data/label.csv");

trainPercent = 0.9;

randIdxs = randperm(size(labelsP, 1));
features = featuresP(randIdxs, :);
labels = labelsP(randIdxs, :);

trainIndex = round(size(labels, 1) * trainPercent);
train_features = features(1:trainIndex, :);
train_labels = labels(1:trainIndex, :);

test_features = features(trainIndex:end, :);
test_labels = labels(trainIndex:end, :);

writematrix(train_features, "classification_features_train.csv");
writematrix(train_labels, "classification_labels_train.csv");

writematrix(test_features, "classification_features_test.csv");
writematrix(test_labels, "classification_labels_test.csv");