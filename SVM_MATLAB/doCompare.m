labels1 = readmatrix("../data/annOutput.csv");
labels2 = readmatrix("../data/svmOutput.csv");

[h,p,ci] = ttest2(labels1, labels2)