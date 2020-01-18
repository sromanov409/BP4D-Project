function [accuracy, bestParameters] = innerfolds(outer_train_features, outer_train_labels, ...
    outer_test_features, outer_test_labels, foldCount, hyperparameters)
    accuracies = zeros(size(hyperparameters, 1), 1);
    
    for i = 1:foldCount
        disp("Inner fold " + i);
        [inner_train_features, inner_train_labels, inner_test_features, inner_test_labels] ...
            = splitFolds(i, foldCount, outer_train_features, outer_train_labels);
        
        accuracies = gridSearch(accuracies, inner_train_features, inner_train_labels, ...
            inner_test_features, inner_test_labels, hyperparameters);
        disp("Accum inner accuracy: " + accuracies(i));
    end
    
    accuracies = accuracies / foldCount;
    [~, index] = max(accuracies);
    
    bestParameters = hyperparameters(index, :);
    
    Mdl = fitcsvm(outer_train_features, outer_train_labels, 'KernelFunction', 'rbf', 'KernelScale', bestParameters{1}, 'BoxConstraint', bestParameters{2});
    accuracy = testModelClassification(Mdl, outer_test_features, outer_test_labels);
    
    
end
