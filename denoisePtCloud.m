function cleaned = denoisePtCloud(ptCloud, numNeighbors, threshold)
%DENOISEPTCLOUD Remove noise from a point cloud.
%   cleaned = denoisePtCloud(ptCloud, numNeighbors, threshold) wraps pcdenoise.
    cleaned = pcdenoise(ptCloud, 'NumNeighbors', numNeighbors, 'Threshold', threshold);
end
