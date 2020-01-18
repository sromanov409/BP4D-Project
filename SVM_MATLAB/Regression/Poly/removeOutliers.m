function [features, labels] =  removeOutliers(features, labels)
    
labelCountPre = length(labels);
[labels, removedInd] = rmoutliers(labels, 'mean', 'ThresholdFactor', 4);
features = features(removedInd == false);
labelCountPost = length(labels);
disp("Data points removed (outliers): " + (labelCountPre-labelCountPost) ...
    + " (" + (100 * ((labelCountPre-labelCountPost) / labelCountPre)) + "%)");