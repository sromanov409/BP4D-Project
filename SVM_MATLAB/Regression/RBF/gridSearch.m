function error = gridSearch(error, trainFeatures, trainLabels, testFeatures, testLabels, hyperparameters)
    for i = 1:length(hyperparameters)
        kernelFunction = hyperparameters{i, 1};
        boxConstraint = hyperparameters{i, 2};
        epsilon = hyperparameters{i, 3};
        disp("Testing parameters: " + kernelFunction + ", " + boxConstraint + ", " + epsilon);
        Mdl = fitrsvm(trainFeatures, trainLabels, 'Standardize', true, 'KernelFunction', 'rbf', 'KernelScale', kernelFunction, 'BoxConstraint', boxConstraint, 'epsilon', epsilon);
        error(i) = error(i) + testModelRegression(Mdl, testFeatures, testLabels);
    end
end