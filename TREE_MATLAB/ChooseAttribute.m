function [best_attribute_idx, best_threshold] = ChooseAttribute(features, labels, thresholdCount)
    best_attribute_idx = 1;
    best_threshold = 0;
    best_gain = intmin;
    for attributeIdx = 1:size(features, 2)
        attributeData = features(:, attributeIdx);
        minVal = min(attributeData);
        maxVal = max(attributeData);
        range = maxVal - minVal;
        step = range / (thresholdCount+1);
        minVal = minVal + step;
        maxVal = maxVal - step;
        
        p = nnz(labels);
        n = length(labels) - p;
        Info = I(p / length(labels), n / length(labels));
        
        for threshold = minVal:step:maxVal
            cond = features(:, attributeIdx) > threshold;
            tp = nnz(cond & labels);
            tn = nnz(~cond & ~labels);
            fp = n - tn;
            fn = p - tp;
            Rem = Remainder(fn, tn, tp, fp);
            
            maxGain = Info - Rem;
            
            %fprintf("gain: %f\n", gain);
            if maxGain > best_gain
                %fprintf("new best gain: %d\n", attributeIdx);
                best_gain = maxGain;
                best_attribute_idx = attributeIdx;
                best_threshold = threshold;
            end
        end
    end
end