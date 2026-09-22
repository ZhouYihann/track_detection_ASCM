function roiCloud = selectROI(ptCloud, roi)
%SELECTROI Keep only the points inside a rectangular region of interest.
%   roi is [xMin xMax; yMin yMax; zMin zMax].
    indices = findPointsInROI(ptCloud, roi);
    roiCloud = select(ptCloud, indices);
end
