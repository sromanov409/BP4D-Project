function error = gridSearch(error, trainFeatures, trainLabels, testFeatures, testLabels, hyperparameters)
    for i = 1:length(hyperparameters)
        order = hyperparameters{i, 1};
        boxConstraint = hyperparameters{i, 2};
        epsilon = hyperparameters{i, 3};
        disp("Testing parameters: " + order + ", " + boxConstraint + ", " + epsilon);
        Mdl = fitrsvm(trainFeatures, trainLabels, 'Standardize', true, 'KernelFunction', 'polynomial', 'PolynomialOrder', order, 'BoxConstraint', boxConstraint, 'epsilon', epsilon, 'Standardize', true);
        error(i) = error(i) + testModelRegression(Mdl, testFeatures, testLabels);
    end
end