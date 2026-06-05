function roiCloud = selectROI(ptCloud, roi)
    indices = findPointsInROI(ptCloud, roi);
    roiCloud = select(ptCloud, indices);
end