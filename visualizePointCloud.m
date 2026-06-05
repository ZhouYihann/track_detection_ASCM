function visualizePointCloud(ptCloud, titleStr)
    figure;
    pcshow(ptCloud);
    title(titleStr);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
end