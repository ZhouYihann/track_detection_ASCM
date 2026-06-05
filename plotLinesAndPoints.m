function plotLinesAndPoints(points, lines)
    hold on;
    scatter3(points(:,1), points(:,2), points(:,3), 30, 'filled');
    for i = 1:size(lines,1)
        lineData = lines(i,:);
        line(lineData([1,4]), lineData([2,5]), lineData([3,6]),...
            'Color','b', 'LineWidth',1.5);
    end
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    hold off;
end