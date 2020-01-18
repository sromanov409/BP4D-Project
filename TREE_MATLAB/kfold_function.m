function [accuracy, recall, precision, fmeasure, output] = kfold_function(k, beta, features, labels, feature_names)

max_depth = 10000000;
n_data = length(labels);

accuracy = 0;
recall = 0;
precision = 0;
fmeasure = 0;

output = ["Fold Number", "Fold Accuracy", "Fold Precision", "Fold Recall", "Fold F-Measure"];

for i = 0:k - 1
    test_start = floor((i * n_data) / k) + 1;
    test_end = floor(((i + 1) * n_data) / k);
    
    disp("test_start: " + test_start)
    disp("test_end: "+ test_end)
    
    test_features = features(test_start:test_end,:);
    test_labels = labels(test_start:test_end,:);
    
    train_features = vertcat(features(1:test_start,:), features(test_end:n_data,:));
    train_labels = vertcat(labels(1:test_start,:), labels(test_end:n_data,:));
    
    tree = makeDecisionTree(train_features, train_labels, feature_names, 0.96, 150, max_depth, 0.0003);
    
    save("tree-for-fold-" + (i + 1) + ".mat", 'tree');
    
    tree_labels = classify_multiple(tree, test_features);
    tree_labels = tree_labels';
    
    true_positives = nnz(test_labels & tree_labels);
    true_negatives = nnz(~test_labels & ~tree_labels);
    false_negatives = nnz(test_labels & ~tree_labels);
    false_positives = nnz(~test_labels & tree_labels);

    fold_recall = true_positives / (true_positives + false_negatives);
    fold_precision = true_positives / (true_positives + false_positives);

    fold_fmeasure = f_measure(beta, fold_recall, fold_precision);

    recall = recall + fold_recall;
    
    testAcc = simpleAccuracy(tree, test_features, test_labels);
    testAcc = round(testAcc, 4);
    
    disp("Fold: " + (i + 1))
    disp("accuracy: " + testAcc)
    disp("precision: " + fold_precision)
    disp("recall: " + fold_recall)
    disp("fmeasure: " + fold_fmeasure)
    
    row = [i + 1, testAcc, fold_precision, fold_recall, fold_fmeasure];
    output = [output; row];
  
    accuracy = accuracy + testAcc;
    precision = precision + fold_precision;
    fmeasure = fmeasure + fold_fmeasure;
end

accuracy = accuracy / k;
recall = recall / k;
precision = precision / k;
fmeasure = fmeasure / k;

row = ["Average", accuracy, recall, precision, fmeasure];
output = [output; row];

end



function measure = f_measure(beta, recall, precision)

beta = beta * beta;
measure = ((1 + beta) * precision * recall) / ((beta * precision) + recall);

end