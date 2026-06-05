clear
close all

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

roi = [0 20; -10 20; 0.0 0.40];
roiCloud = selectROI(ptCloud, roi);
visualizePointCloud(roiCloud, 'ROI Filtered Point Cloud');
view(3);         
grid on;

if ~isempty(centerline)
    hold on;
    plot3(centerline(:,1), centerline(:,2), centerline(:,3), 'r-', 'LineWidth', 2);
    hold off;
end

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