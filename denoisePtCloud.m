function cleaned = denoisePtCloud(ptCloud, numNeighbors, threshold)
    cleaned = pcdenoise(ptCloud, 'NumNeighbors', numNeighbors, 'Threshold', threshold);
end