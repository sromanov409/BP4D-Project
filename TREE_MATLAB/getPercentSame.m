function [percent, value] = getPercentSame(list)
    ones = nnz(list);
    value = 1;
    if (ones/length(list)) < 0.5
        ones = length(list) - ones;
        value = 0;
    end
    percent = ones / length(list);
end