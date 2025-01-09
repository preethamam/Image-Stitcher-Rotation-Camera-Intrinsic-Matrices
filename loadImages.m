function [images, imageSizes, imageNames, numImgs] = loadImages(imgFolder)
    % LOADIMAGES Load and preprocess images from a specified folder.
    %   [images, imageSizes, imageNames, numImgs] = LOADIMAGES(imgFolder)
    %   reads all images from the folder specified by imgFolder, extracts
    %   their sizes, names, and returns the images in a cell array.
    %
    %   Inputs:
    %       imgFolder - String specifying the folder containing images.
    %
    %   Outputs:
    %       images - Cell array containing the loaded images.
    %       imageSizes - Matrix containing the sizes of the images.
    %       imageNames - Cell array containing the names of the images.
    %       numImgs - Number of images in the folder.

    % Read images
    % Create an imageDatastore to manage the collection of images in the folder
    imds = imageDatastore(imgFolder);
    % Read all images from the datastore into a cell array
    imageFiles = readall(imds);    
    % Extract the names and extensions of the image files
    [~,imageNames,ext] = fileparts(imds.Files);
    imageNames = strcat(imageNames,ext);

    % Number of images in the folder
    numImgs = length(imageFiles);

    % Initialize the cell arrays
    % Create a cell array to hold the images
    images = cell(1,numImgs);
    
    % Image extraction
    % Use a parallel for loop to process each image
    parfor i = 1:numImgs
        % Sequential mages
        % Read the current image from the cell array
        image = imageFiles{i};
               
        % Get size of the image
        % Determine the size of the current image
        [imRows, imCols, imChannel] = size(image);
        % Store the size of the current image in the imageSizes matrix
        imageSizes(i,:) = [imRows, imCols, imChannel];
        
        % Replicate the third channel
        % If the image is grayscale (single channel), replicate it to create an RGB image
        if imChannel == 1
            image = repmat(image, 1, 1, 3);
        end

        % Stack images
        % Store the processed image in the cell array
        images{i} = image;        
    end
end