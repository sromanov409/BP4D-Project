function data = untranslateYaw(data)
    data(data < 0) = (data(data < 0)) + 360;
    data = 180 - data;
end