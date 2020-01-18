function [error, bestParameters] = innerfolds(outer_train_features, outer_train_labels, ...
    outer_test_features, outer_test_labels, foldCount, hyperparameters)
    errors = zeros(size(hyperparameters, 1), 1);
    
    for i = 1:foldCount
        disp("Inner fold " + i);
        [inner_train_features, inner_train_labels, inner_test_features, inner_test_labels] ...
            = splitFolds(i, foldCount, outer_train_features, outer_train_labels);
        
        errors = gridSearch(errors, inner_train_features, inner_train_labels, ...
            inner_test_features, inner_test_labels, hyperparameters);
        disp("Accum inner error: " + errors(i));
    end
    
    errors = errors / foldCount;
    [~, index] = min(errors);
    
    bestParameters = hyperparameters(index, :);
    
    Mdl = fitrsvm(outer_train_features, outer_train_labels, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', bestParameters{1}, 'BoxConstraint', bestParameters{2}, 'epsilon', bestParameters{3});
    error = testModelRegression(Mdl, outer_test_features, outer_test_labels);
    
    
end
