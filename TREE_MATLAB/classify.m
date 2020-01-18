function class = classify(tree, features)
    if isempty(tree.kids)
        class = tree.class;
    else
        if features(tree.attribute) < tree.threshold
            class = classify(tree.kids{1,1}, features);
        else
            class = classify(tree.kids{1,2}, features);
        end
    end
end