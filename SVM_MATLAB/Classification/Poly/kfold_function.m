function output = kfold_function(k, features, labels, kernel_function, params)
n_data = length(labels);

accuracy = 0;
recall = 0;
precision = 0;
fmeasure = 0;
support_vectors_count = 0;
support_vectors_percent = 0;

output = ["Fold Number", "Fold Accuracy", "Fold Precision", "Fold Recall", "Fold F-Measure", "Support Vector Count", "Support Vector Percent"];

for i = 0:k - 1
    test_start = floor((i * n_data) / k) + 1;
    test_end = floor(((i + 1) * n_data) / k);
    
    disp("test_start: " + test_start)
    disp("test_end: "+ test_end)
    
    test_features = features(test_start:test_end,:);
    test_labels = labels(test_start:test_end,:);
    
    train_features = vertcat(features(1:test_start,:), features(test_end:n_data,:));
    train_labels = vertcat(labels(1:test_start,:), labels(test_end:n_data,:));
    
    svm = fitcsvm(train_features, train_labels, 'KernelFunction', kernel_function, 'PolynomialOrder', params{1}, 'BoxConstraint', params{2}, 'Standardize', true);
    
    testAcc = testModelClassification(svm, test_features, test_labels);
    testAcc = round(testAcc, 4);

    tree_labels = predict(svm, test_features);
    
    tree_labels = tree_labels';
    
    true_positives = nnz(test_labels & tree_labels);
    true_negatives = nnz(~test_labels & ~tree_labels);
    false_negatives = nnz(test_labels & ~tree_labels);
    false_positives = nnz(~test_labels & tree_labels);

    fold_recall = true_positives / (true_positives + false_negatives);
    fold_precision = true_positives / (true_positives + false_positives);

    fold_fmeasure = f_measure(1, fold_recall, fold_precision);
    fold_support_vectors_count = size(svm.SupportVectors, 1);
    fold_support_vectors_percent = fold_support_vectors_count / length(train_labels);
    recall = recall + fold_recall;
        
    disp("Fold: " + (i + 1))
    disp("accuracy: " + testAcc)
    disp("precision: " + fold_precision)
    disp("recall: " + fold_recall)
    disp("fmeasure: " + fold_fmeasure)

    
    row = [i + 1, testAcc, fold_precision, fold_recall, fold_fmeasure, fold_support_vectors_count, fold_support_vectors_percent];
    output = [output; row];
  
    accuracy = accuracy + testAcc;
    precision = precision + fold_precision;
    fmeasure = fmeasure + fold_fmeasure;
    support_vectors_count = support_vectors_count + fold_support_vectors_count;
    support_vectors_percent = support_vectors_percent + fold_support_vectors_percent;
end

accuracy = accuracy / k;
recall = recall / k;
precision = precision / k;
fmeasure = fmeasure / k;
support_vectors_count = support_vectors_count / k;
support_vectors_percent = support_vectors_percent / k;

row = ["Average", accuracy, recall, precision, fmeasure, support_vectors_count, support_vectors_percent];
output = [output; row];

end

function measure = f_measure(beta, recall, precision)

beta = beta * beta;
measure = ((1 + beta) * precision * recall) / ((beta * precision) + recall);

end