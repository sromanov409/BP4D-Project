function accuracy = testModelClassification(Mdl, testFeatures, testLabels)
    [predictions, ~] = predict(Mdl, testFeatures);

    correctCount = nnz(predictions == testLabels);
    accuracy = correctCount / length(testLabels);
end