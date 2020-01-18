function tree = makeDecisionTree(features, labels, featureNames, ...
    stopPercent, thresholdCount, maxDepth, labelCountPercen, depth, labelCount)

    if nargin < 8
       depth = 0;
       labelCount = length(features);
    end
    tree = struct('op', 0, 'class', 0, 'attribute', '', 'threshold', 0);
    tree.kids = {};
    [same, value] = getPercentSame(labels);
    if same >= stopPercent || depth > maxDepth || length(labels) < (labelCount * labelCountPercen)
        tree.class = value;
    else
        [attribute, threshold] = ChooseAttribute(features, labels, thresholdCount);
        tree.op = featureNames(attribute);
        tree.attribute = attribute;
        tree.threshold = threshold;
        tree.kids = cell(1,2);

        left = features(:,attribute) < threshold;
        tree.kids{1,1} = makeDecisionTree(features(left,:), labels(left), featureNames, stopPercent, thresholdCount, maxDepth, labelCountPercen, depth+1, labelCount);
        tree.kids{1,2} = makeDecisionTree(features(~left,:), labels(~left), featureNames, stopPercent, thresholdCount, maxDepth, labelCountPercen, depth+1, labelCount);
        
    end
end