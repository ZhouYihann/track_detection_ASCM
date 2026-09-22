%MAIN_DEMO Demo of the track-side line-fitting pipeline on a synthetic ribbon cloud.
%   Load a PLY point cloud, crop to a ROI, slice along the track, fit short
%   line segments to each slice, and plot the result.

clear
close all

% Add the script's folder to the path so helper functions are found.
scriptDir = fileparts(mfilename('fullpath'));
if ~isempty(scriptDir)
    addpath(scriptDir, '-begin');
    rehash path;
end

plyFile = fullfile(scriptDir, 'syntheticRibbonCloud.ply');
if ~exist(plyFile, 'file')
    error('PLY file not found: %s. Please generate it first with exportSyntheticRibbonPLY.', plyFile);
end

ptCloud = pcread(plyFile);
centerline = [];
visualizePointCloud(ptCloud, 'Synthetic Point Cloud from PLY');

roi = [0 20; -10 20; 0.0 0.40];   % [xMin xMax; yMin yMax; zMin zMax]
roiCloud = selectROI(ptCloud, roi);
visualizePointCloud(roiCloud, 'ROI Filtered Point Cloud');
view(3);
grid on;

if ~isempty(centerline)
    hold on;
    plot3(centerline(:,1), centerline(:,2), centerline(:,3), 'r-', 'LineWidth', 2);
    hold off;
end

% Fit 0.84-length segments along the x-axis, one per 0.25-thick slice.
sliceInterval = 0.25;
[leftDirectionVectors, leftPoints, leftLines] = processTrackSide(roiCloud, sliceInterval, ptCloud, 'left', [1, 0, 0], 0.84);

allPoints = [leftPoints];
allLines = [leftLines];

figure('Name', 'Synthetic Demo Final Result', 'Color', 'w');
plotLinesAndPoints(allPoints, allLines);
hold on;
legend('fitting points','fitting lines', 'Location', 'best');
title('Synthetic Demo Result vs. True Centerline');
hold off;
