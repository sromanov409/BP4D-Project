function information = I(pPercen, nPercen)
    if isnan(pPercen)
        information = 0;
    else
        if pPercen == 1 || nPercen == 1
            information = 0;
        else
            information = -pPercen*log2(pPercen) - nPercen*log2(nPercen);
        end
    end
end