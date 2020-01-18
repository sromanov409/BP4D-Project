function value = Gain(features, labels, attributeIdx, threshold, useLT)
    p = nnz(labels);
    n = length(labels) - p;
    
    if useLT
        cond = features(:, attributeIdx) < threshold;
    else
        cond = features(:, attributeIdx) > threshold;
    end
    
    tp = nnz(cond & labels);
    tn = nnz(~cond & ~labels);
    fp = n - tn;
    fn = p - tp;
    
    pPercen = p / (p + n);
    nPercen = n / (p + n);
    
    Info = I(pPercen, nPercen);
    Rem  = Remainder(fn, tn, tp, fp);
    value = Info - Rem;
end