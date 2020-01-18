function prod = cartProd(param1, param2)
    prod = cell(size(param1, 1) * size(param2, 1));
    index = 1;
    for i = 1 : length(param1)
        for j = 1 : length(param2)
            prod{index, 1} = param1(i);
            prod{index, 2} = param2(j);
            index = index + 1;
        end
    end
end
