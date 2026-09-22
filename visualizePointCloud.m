function visualizePointCloud(ptCloud, titleStr)
%VISUALIZEPOINTCLOUD Display a point cloud in a new figure.
    figure;
    pcshow(ptCloud);
    title(titleStr);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
end
