function accuracy = gridSearch(accuracy, trainFeatures, trainLabels, testFeatures, testLabels, hyperparameters)
    for i = 1:length(hyperparameters)
        kernelFunction = hyperparameters{i, 1};
        boxConstraint = hyperparameters{i, 2};
        disp("Testing parameters: " + kernelFunction + ", " + boxConstraint);
        Mdl = fitcsvm(trainFeatures, trainLabels, 'KernelFunction', 'rbf', 'KernelScale', kernelFunction, 'BoxConstraint', boxConstraint);
        accuracy(i) = accuracy(i) + testModelClassification(Mdl, testFeatures, testLabels);
    end
end