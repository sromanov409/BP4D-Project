function a = simpleAccuracy(tree, features, labels)
    correct = 0;
    for i = 1:length(features)
        if classify(tree, features(i, :)) == labels(i)
            correct = correct + 1;
        end
    end
    a = correct / length(features);

end