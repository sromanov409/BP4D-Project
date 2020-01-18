function output = kfold_function(k, features, labels, kernelFunction, params)

	n_data = length(labels);

	totalError = 0;
	totalSVs = 0;
	totalSVPercent = 0;

	output = ["Fold Number", "Fold error", "Support vector count", "Support vector percent"];

	for i = 0:k - 1
	    test_start = floor((i * n_data) / k) + 1;
	    test_end = floor(((i + 1) * n_data) / k);
	    disp("Fold: "+(i+1));
	    disp("test_start: " + test_start)
	    disp("test_end: "+ test_end)
	    
	    test_features = features(test_start:test_end,:);
	    test_labels = labels(test_start:test_end,:);
	    
	    train_features = vertcat(features(1:test_start,:), features(test_end:n_data,:));
	    train_labels = vertcat(labels(1:test_start,:), labels(test_end:n_data,:));
	    
	    Mdl = fitrsvm(train_features, train_labels, 'Standardize', true, 'KernelFunction', kernelFunction, 'KernelScale', params{1}, 'BoxConstraint', params{2}, 'epsilon', params{3});
	    
	    curError = testModelRegression(Mdl, test_features, test_labels)
	    SVCount = size(Mdl.SupportVectors, 1);

	    SVPercent = SVCount / length(train_labels);

	    totalSVs = totalSVs + SVCount;
	    totalError = totalError + curError;
	    totalSVPercent = totalSVPercent + SVPercent;

	    row = [i + 1, curError, SVCount, SVPercent];
	    output = [output; row];
	  
	end

	row = ["Average", totalError / k, totalSVs / k, totalSVPercent / k];
	output = [output; row];

end


