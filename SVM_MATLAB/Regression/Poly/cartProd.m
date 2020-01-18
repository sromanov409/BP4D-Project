function prod = cartProd(param1, param2, param3)
    prod = cell(size(param1, 1) * size(param2, 1) * size(param3, 1));
    index = 1;
    for i = 1 : length(param1)
        for j = 1 : length(param2)
            for k = 1 : length(param3)
                prod{index, 1} = param1(i);
                prod{index, 2} = param2(j);
                prod{index, 3} = param3(k);
                index = index + 1;
            end
        end
    end
end
