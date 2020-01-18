function rem = Remainder(p1,n1,p2,n2)
    sum = p1 + p2 + n1 + n2;
    rem1 = ((p1+n1) / sum) * I(p1 / (p1 + n1), n1 / (p1 + n1));
    rem2 = ((p2+n2) / sum) * I(p2 / (p2 + n2), n2 / (p2 + n2));
    rem = rem1 + rem2;
end