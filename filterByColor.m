function filtered = filterByColor(ptCloud, colorThresholds, threshold)
    colors = ptCloud.Color;
    if isa(colors(1,1), 'uint8')
        colors = double(colors)/255;
    end
    
    mask = false(size(colors,1),1);
    for i = 1:size(colorThresholds,1)
        diff = sqrt(sum((colors - colorThresholds(i,:)).^2, 2));
        mask = mask | (diff < threshold);
    end
    filtered = select(ptCloud, find(mask));
end