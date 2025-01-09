%//%*****************************************************************************%
%//%*                 Automatic Panorama Image Stitcher                         *%
%//%*   Stitches panorama image given the Rotation and camera intrinsic         *%
%//%*                 matrices after the bundle adjustment                      *%
%//%*                    Name: Dr. Preetham Manjunatha                          *%
%//%*               GitHub: https://github.com/preethamam	                    *%
%//%*             Repo Name: AutoPanoStitch (auxiliary function)                *%
%//%*                    Written Date: 01/08/2025                               *%
%%**************************************************************************    *%
%* Citation 1: Automatic Panoramic Image Stitching using Invariant Features.    *% 
%* M. Brown and D. Lowe. International Journal of Computer Vision. 74(1),       *%
%* pages 59-73, 2007                                                            *%
%*                                                                              *%
%* Citation 2: Recognising Panoramas. M. Brown and D. G. Lowe.                  *%
%* International Conference on Computer Vision (ICCV2003). pages 1218-1225,     *%
%* Nice, France, 2003.                                                          *%
%*                                                                              *%
%* Please refer to the GitHub repo:                                             *%
%* https://github.com/preethamam/AutomaticPanoramicImageStitching-AutoPanoStitch*%
%* for the full Automatic Panorama Image Stitcher (AutoPanoStitch).             *%
%********************************************************************************%

%% Start
%--------------------------------------------------------------------------
clear; close all; clc;
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
warning('off','all');
Start = tic;

%% Get inputs
%--------------------------------------------------------------------------
% Inputs file
%--------------------------------------------------------------------------
blendFeather = 0;
imagesFolder = 'images';

%--------------------------------------------------------------------------
% Load files
%--------------------------------------------------------------------------
load cameras.mat

%% Read images
[images, imageSizes, imageNames, numImgs] = loadImages(imagesFolder);

%% Stitch images
customrender = tic;
panorama = stitchImages_tightROI(images, cameras, blendFeather);
fprintf('Custom render: %f seconds\n', toc(customrender));

%% Show panorama
figure; 
imshow(panorama)

%% End parameters
%--------------------------------------------------------------------------
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
statusFclose = fclose('all');
if(statusFclose == 0)
    disp('All files are closed.')
end
Runtime = toc(Start);
fprintf('Total runtime : %f seconds\n', Runtime);
currtime = datetime('now');
display(currtime)