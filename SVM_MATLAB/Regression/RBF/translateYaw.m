function data = translateYaw(data)
    data = 180 - data;
    data(data > 180) = (data(data > 180)) - 360;
end

