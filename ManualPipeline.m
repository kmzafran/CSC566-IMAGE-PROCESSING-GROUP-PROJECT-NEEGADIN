% ManualPipeline.m
% Automated Plant Detection and Segmentation
% CSC566 Image Processing Project
%
% This script implements a manual pipelines using methods specified in the
% project guidelines:
% 1. Pre-processing (Noise Reduction, Contrast Enhancement) - Guide 4.i
% 2. Color Segmentation (Histogram/Color-based) - Guide 8.a
% 3. Thresholding (Otsu) - Guide 4.a
% 4. Morphological Operations - Guide 3.d
% 5. Morphological Operations - Guide 3.d

clc;
clear;
close all;

%% 1. Configuration and Image Loading
% Load the specific sample image '1.png' from the current directory
imageFilename = '5.png';

if ~exist(imageFilename, 'file')
    error('Image file "%s" not found. Please ensure it is in the same folder as this script.', imageFilename);
end

originalImage = imread(imageFilename);

figure('Name', 'Step 1: Original Image');
imshow(originalImage);
title('Original Aerial Image');

%% 2. Pre-processing (Guide 4.i)
% a) Noise Reduction using Gaussian Smoothing
sigma = 2;
denoisedImage = imgaussfilt(originalImage, sigma);

% b) Contrast Enhancement using CLAHE (on L channel of LAB)
% Convert to LAB color space
labImage = rgb2lab(denoisedImage);
L_channel = labImage(:,:,1);

% Apply CLAHE to L channel (scale to 0-1 range for adapthisteq if needed)
L_channel = L_channel / 100; % LAB L is 0-100
L_enhanced = adapthisteq(L_channel, 'NumTiles', [8 8], 'ClipLimit', 0.01);
L_enhanced = L_enhanced * 100;

labImage(:,:,1) = L_enhanced;
enhancedImage = lab2rgb(labImage);

figure('Name', 'Step 2: Pre-processing');
subplot(1,2,1); imshow(denoisedImage); title('Denoised (Gaussian)');
subplot(1,2,2); imshow(enhancedImage); title('Contrast Enhanced (CLAHE)');

%% 3. Color Space Analysis & Vegetation Indices (Guide 8.a, 4.iii)
% We used the "Excess Green" (ExG) index which is standard for plant detection
% ExG = 2*G - R - B

doubleImg = im2double(enhancedImage);
R = doubleImg(:,:,1);
G = doubleImg(:,:,2);
B = doubleImg(:,:,3);

% Calculate Excess Green Index
ExG = 2*G - R - B;

% Normalize ExG for visualization and thresholding
ExG_norm = (ExG - min(ExG(:))) / (max(ExG(:)) - min(ExG(:)));

figure('Name', 'Step 3: Vegetation Index');
imshow(ExG_norm);
colormap(gca, 'jet');
colorbar;
title('Excess Green (ExG) Index Map');

%% 4. Thresholding (Guide 4 - Otsu method)
% Apply global thresholding using Otsu's method
level = graythresh(ExG_norm) * 0.8; % Lower threshold to capture more plants
binaryMask = imbinarize(ExG_norm, level);
binaryMask = medfilt2(binaryMask, [5 5]); % Smoothen rough edges

figure('Name', 'Step 4: Otsu Thresholding');
imshow(binaryMask);
title(['Otsu Thresholding (Level = ' num2str(level, '%.2f') ')']);

%% 5. Morphological Operations (Guide 3.d)
% Clean up the binary mask

% a) Remove small objects (noise) using Area Open
minPlantArea = 50; % Reduced to keep smaller plants
binaryMask = bwareaopen(binaryMask, minPlantArea);

% b) Open operation: Remove connections between rows
% Relaxed SE to avoid erasing small plants
se_open_rect = strel('rectangle', [2 3]); 
cleanMask = imopen(binaryMask, se_open_rect);

% Also apply a disk open to smooth jagged edges
se_open_disk = strel('disk', 1);
cleanMask = imopen(cleanMask, se_open_disk);

% c) Close operation to round out the blobs
se_close = strel('disk', 2);
cleanMask = imclose(cleanMask, se_close);

% d) Shape-based filtering: Remove elongated blobs (likely soil strips)
% Plants from above tend to be roughly circular, soil strips are elongated
props = regionprops(cleanMask, 'Eccentricity', 'PixelIdxList');
for i = 1:length(props)
    % Eccentricity ranges from 0 (circle) to 1 (line)
    % Remove highly elongated shapes (eccentricity > 0.9)
    if props(i).Eccentricity > 0.9
        cleanMask(props(i).PixelIdxList) = 0;
    end
end

figure('Name', 'Step 5: Morphological Processing');
imshow(cleanMask);
title('Morphologically Cleaned Mask');


%% 6. Visualization Matching User References

% Class Segmentation (Green blobs on Black background)
% Create an RGB image where the mask is Green (0, 255, 0) and background is Black
classSegColors = zeros(size(cleanMask, 1), size(cleanMask, 2), 3);
classSegColors(:,:,2) = cleanMask; % Set Green channel
classSegViz = im2double(classSegColors); 

figure('Name', 'Final Results');
imshow(classSegViz);
title('Class Segmentation (Matched to Reference)');
