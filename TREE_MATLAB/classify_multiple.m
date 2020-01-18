function labels = classify_multiple(tree, features)
    labels = zeros(1, length(features));
    for i = 1:length(features)
       featureRow = features(i, :);
       label = classify(tree, featureRow);
       labels(i) = label;
    end
end