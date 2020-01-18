function [train_features, train_labels, test_features, test_labels] = splitFolds(i, foldCount, features, labels)
    n_data = length(labels);
    
    i = i - 1;
    test_start = floor((i * n_data) / foldCount) + 1;
    test_end = floor(((i + 1) * n_data) / foldCount);

    test_features = features(test_start:test_end,:);
    test_labels = labels(test_start:test_end,:);

    train_features = vertcat(features(1:test_start,:), features(test_end:end,:));
    train_labels = vertcat(labels(1:test_start,:), labels(test_end:end,:));
end
