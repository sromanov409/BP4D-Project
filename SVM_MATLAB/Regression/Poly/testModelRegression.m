function error = testModelRegression(Mdl, features, labels)
    predictions = predict(Mdl, features);
    
    error = sqrt(mean((labels-predictions).^2));
end