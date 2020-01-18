function accuracy = gridSearch(accuracy, trainFeatures, trainLabels, testFeatures, testLabels, hyperparameters)
    for i = 1:length(hyperparameters)
        order = hyperparameters{i, 1};
        boxConstraint = hyperparameters{i, 2};
        disp("Testing parameters: " + order + ", " + boxConstraint);
        Mdl = fitcsvm(trainFeatures, trainLabels, 'KernelFunction', 'polynomial', 'PolynomialOrder', order, 'BoxConstraint', boxConstraint, 'Standardize', true);
        accuracy(i) = accuracy(i) + testModelClassification(Mdl, testFeatures, testLabels);
    end
end