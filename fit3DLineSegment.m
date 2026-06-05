function [startPt, endPt, midPt] = fit3DLineSegment(projPoints3D, segmentLength, initialMidPt, lines)

validateattributes(projPoints3D, {'numeric'}, {'ncols',3,'2d'});
validateattributes(segmentLength, {'numeric'}, {'scalar','positive'});
projPoints3D = double(projPoints3D); 

if size(projPoints3D, 1) < 2
    error('fit3DLineSegment:NotEnoughPoints', 'At least two points are needed for line segment fitting.');
end

if nargin < 3 || isempty(initialMidPt)
    midPt_initial = mean(projPoints3D, 1);
else
    validateattributes(initialMidPt, {'numeric'}, {'vector','numel',3});
    midPt_initial = double(initialMidPt(:)');
end
centeredPoints = projPoints3D - midPt_initial;

if all(vecnorm(centeredPoints, 2, 2) < 1e-12)
    mainDirection = [1, 0, 0];
else
    [~, ~, V] = svd(centeredPoints, 'econ');
    mainDirection = double(V(:, 1)'); 
end

t = (projPoints3D - midPt_initial) * mainDirection'; 

L = segmentLength;
objectiveFunc = @(delta) sum(arrayfun(@(ti) computeDistanceSquared(ti - delta, L), t, 'UniformOutput', true));
options = optimset('Display', 'off');
delta_opt = fminsearch(objectiveFunc, 0, options);

midPt = midPt_initial + delta_opt * mainDirection;
halfVector = (segmentLength/2) * mainDirection;
startPt = midPt - halfVector;
endPt = midPt + halfVector;

calcLength = norm(endPt - startPt);
if abs(calcLength - segmentLength) > 1e-6
    error('Line segment length error exceeds tolerance: calculated value %.6f vs specified value %.6f', calcLength, segmentLength);
end

function dSq = computeDistanceSquared(t_proj, L)
    t_proj = double(t_proj); 
    if t_proj < -L/2
        d = -L/2 - t_proj;
    elseif t_proj > L/2
        d = t_proj - L/2;
    else
        d = 0;
    end
    dSq = d^2;
end

if mod(size(lines, 1), 10) == 0
    figure;
    scatter3(projPoints3D(:,1), projPoints3D(:,2), projPoints3D(:,3), 'b.');
    hold on;
    plot3([startPt(1), endPt(1)], [startPt(2), endPt(2)], [startPt(3), endPt(3)],...
          'r-', 'LineWidth', 2);
    scatter3(midPt_initial(1), midPt_initial(2), midPt_initial(3), 'mo', 'filled');
    scatter3(midPt(1), midPt(2), midPt(3), 'go', 'filled');
    axis equal;
    title(sprintf('3D line segment fitting (L=%.2f)', segmentLength));
    xlabel('X'); ylabel('Y'); zlabel('Z');
    legend('Projection point', 'Fitted line segment', 'Initial midpoint', 'Optimized midpoint', 'Location', 'best');
end
end