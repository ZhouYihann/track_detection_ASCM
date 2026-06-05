function [directions, averagePoints, lines] = processTrackSide(cloud, sliceInterval, ~, ~, direction, L)
    direction = direction / norm(direction);

    averagePoints = [];
    directions = [];
    lines = [];
    initialMidPt = [];

    projVals = cloud.Location * direction';
    minProj = min(projVals);
    p_end = (minProj + sliceInterval) * direction; 
    %if size(averagePoints, 1) > 2
        %p_end = initialMidPt
   while true
        directions = [directions; direction];
        d = cloud.Location - p_end;
        distToPlane = d * direction';
        
        filteredPoints = cloud.select(distToPlane >= -sliceInterval & distToPlane < 0);
   
        if isempty(filteredPoints.Location)
            break;
        end
    
        points = double(filteredPoints.Location);
        
        if isempty(initialMidPt)
            planePoint = p_end;
        else
            planePoint = initialMidPt;
        end
        
        projPoints = points - ((points - planePoint) * direction') * direction;

        if size(projPoints, 1) < 2 || size(unique(projPoints, 'rows'), 1) < 2
            break
        end
        [start, endPt, mid] = fit3DLineSegment(projPoints, L, initialMidPt, lines);
        lines = [lines; start,endPt];
        averagePoints = [averagePoints; mid];
        
        if size(averagePoints, 1) > 2
            newDir = diff(averagePoints(end-1:end, :));

            if norm(newDir) > eps
                newDir = newDir / norm(newDir);
                if dot(newDir, direction) < 0
                    newDir = -newDir;
                end
                direction = newDir;
            end
        end
        initialMidPt = averagePoints(end, :) + sliceInterval * direction;
        p_end = p_end + sliceInterval * direction;
    end
end