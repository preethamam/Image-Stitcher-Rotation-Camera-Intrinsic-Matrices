function stitched = stitchImages_tightROI(images, cameras, blendFeather, ref_idx)
    % STITCHIMAGES_TIGHTROI Stitch multiple images into a panorama.
    %   stitched = STITCHIMAGES_TIGHTROI(images, cameras, ref_idx, blendFeather)
    %   stitches the input images into a single panorama using the provided
    %   camera parameters and reference image index with an automatic bounds 
    %   detected using the homographies.
    %
    %   Faster than the STITCHIMAGES_LARGEROI ~ 0.7000 seconds for the given
    %   images
    %
    %   Inputs:
    %       images - Cell array containing the input images.
    %       cameras - Struct array containing camera parameters (K and R).
    %       blendFeather - Flag to enable/disable feather blending (0 or 1).
    %       ref_idx - Index of the reference image.
    %
    %   Outputs:
    %       stitched - The resulting stitched panorama image.
    
    % Find the identity matrix index
    if nargin < 4
        cameras_R = {cameras.R};
        ref_idx = find(cellfun(@(x) sum(sum(x - eye(size(x,1)))), cameras_R) == 0);
        if isempty(ref_idx)
            ref_idx = 1;
        end
    end

    % Get reference camera R and K
    camRefK = cameras(ref_idx).K;
    camRefR = cameras(ref_idx).R;

    % Get reference image dimensions
    [h, w, ~] = size(images{ref_idx});    
    
    % Sample more points along the image boundaries for better bound estimation
    [X, Y] = meshgrid([1 w/4 w/2 3*w/4 w], [1 h/4 h/2 3*h/4 h]);
    points = [X(:)'; Y(:)'; ones(1, numel(X))];
    
    % Initialize bounds
    min_x = inf; max_x = -inf;
    min_y = inf; max_y = -inf;
    
    % Project points from all images
    parfor i = 1:length(images)
        H = (camRefK * camRefR * cameras(i).R') / cameras(i).K;
        projected = H * points;
        projected = bsxfun(@rdivide, projected, projected(3,:));
        
        min_x = min(min_x, min(projected(1,:)));
        max_x = max(max_x, max(projected(1,:)));
        min_y = min(min_y, min(projected(2,:)));
        max_y = max(max_y, max(projected(2,:)));
    end
    
    % Add small margin
    margin = 25;
    output_w = ceil(max_x - min_x) + 2*margin;
    output_h = ceil(max_y - min_y) + 2*margin;
    output_view = [output_h, output_w];
    
    % Create translation matrix to ensure all points are positive
    T = [1 0 -min_x+margin;
         0 1 -min_y+margin;
         0 0 1];

    % Initialize arrays for warped images and masks
    warped_images = cell(length(images), 1);
    masks = cell(length(images), 1);
    
    % First pass: warp all images
    parfor i = 1:length(images)
        
        % Compute homography from reference to current view
        H = (T * camRefK * camRefR * cameras(i).R') / cameras(i).K;
        
        % Warp image
        tform = H';
        warped = imWarp(images{i}, tform, output_view);
        
        % Create mask
        mask = any(warped > 0, 3);
        
        warped_images{i} = warped;
        masks{i} = double(mask);
    end
        
    % Simple feathering at boundaries
    switch blendFeather
        case 0
            % Blend images directly without feathering
            accumulated = zeros(output_h, output_w, 3, 'double');
            weight_sum = zeros(output_h, output_w, 'double');
            for i = 1:length(images)
                curr_mask = repmat(masks{i}, [1 1 3]);
                accumulated = accumulated + double(warped_images{i}) .* curr_mask;
                weight_sum = weight_sum + masks{i};
            end
            
            % Normalize
            weight_sum(weight_sum < eps) = 1;
            weight_sum = repmat(weight_sum, [1 1 3]);
            stitched = uint8(accumulated ./ weight_sum);
        case 1
            % Blend images directly with feathering
            kernel_size = 3;
            kernel = fspecial('gaussian', [kernel_size kernel_size], kernel_size/6);
            
            for i = 1:length(masks)
                masks{i} = imfilter(masks{i}, kernel, 'replicate');
            end
            
            % Blend images
            accumulated = zeros(output_h, output_w, 3, 'double');
            weight_sum = zeros(output_h, output_w, 'double');
            
            for i = 1:length(images)
                curr_mask = repmat(masks{i}, [1 1 3]);
                accumulated = accumulated + double(warped_images{i}) .* curr_mask;
                weight_sum = weight_sum + masks{i};
            end
            
            % Normalize
            weight_sum(weight_sum < eps) = 1;
            weight_sum = repmat(weight_sum, [1 1 3]);
            stitched = uint8(accumulated ./ weight_sum);        
    end
    
    % Crop black borders
    [rows, cols] = find(rgb2gray(stitched) > 0);
    if ~isempty(rows) && ~isempty(cols)
        min_row = max(1, min(rows) - 10);
        max_row = min(output_h, max(rows) + 10);
        min_col = max(1, min(cols) - 10);
        max_col = min(output_w, max(cols) + 10);
        stitched = stitched(min_row:max_row, min_col:max_col, :);
    end
end

function warped = imWarp(image, tform, output_view, options)
    % Highly vectorized implementation of image warping
    % Args:
    %   img: Input image (HxWx3 uint8)
    %   tform: projective2d transform object
    %   output_view: imref2d object defining output limits
    %   options: struct with fields:
    %     - method: 'nearest', 'bilinear', or 'bicubic'
    %     - fill_value: value for outside pixels (default: 0)
    
    % Default options
    if nargin < 4, options = struct(); end
    if ~isfield(options, 'method'), options.method = 'bilinear'; end
    if ~isfield(options, 'fill_value'), options.fill_value = 0; end
    
    % Get dimensions
    [out_height, out_width] = deal(output_view(1), output_view(2));
    [in_height, in_width, num_channels] = size(image);
    total_pixels = out_height * out_width;
    
    % Pre-compute source coordinates
    [X, Y] = meshgrid(single(1:out_width), single(1:out_height));
    src_coords = reshape(tform' \ [X(:)'; Y(:)'; ones(1, total_pixels, 'single')], 3, []);
    src_x = reshape(src_coords(1,:) ./ src_coords(3,:), out_height, out_width);
    src_y = reshape(src_coords(2,:) ./ src_coords(3,:), out_height, out_width);
    
    % Initialize output
    warped = zeros([out_height, out_width, num_channels], 'uint8') + options.fill_value;
    
    switch lower(options.method)
        case 'nearest'
            % Round coordinates and create mask
            x = round(src_x);
            y = round(src_y);
            valid = x >= 1 & x <= in_width & y >= 1 & y <= in_height;
            
            % Convert to linear indices
            indices = sub2ind([in_height, in_width], y(valid), x(valid));
            valid_linear = valid;
            
            % Process all channels at once using reshaping
            img_reshaped = reshape(image, [], num_channels);
            warped_reshaped = reshape(warped, [], num_channels);
            warped_reshaped(valid_linear, :) = img_reshaped(indices, :);
            warped = reshape(warped_reshaped, out_height, out_width, num_channels);
            
        case 'bilinear'
            % Floor coordinates and compute weights
            x1 = floor(src_x);
            y1 = floor(src_y);
            x2 = x1 + 1;
            y2 = y1 + 1;
            
            wx = src_x - x1;
            wy = src_y - y1;
            
            % Find valid coordinates
            valid = x1 >= 1 & x2 <= in_width & y1 >= 1 & y2 <= in_height;
            valid_linear = find(valid);
            
            if ~isempty(valid_linear)
                % Get corner indices for valid pixels
                i11 = sub2ind([in_height, in_width], y1(valid), x1(valid));
                i12 = sub2ind([in_height, in_width], y2(valid), x1(valid));
                i21 = sub2ind([in_height, in_width], y1(valid), x2(valid));
                i22 = sub2ind([in_height, in_width], y2(valid), x2(valid));
                
                % Extract weights for valid pixels
                wx = wx(valid);
                wy = wy(valid);
                
                % Compute weights
                w11 = (1-wx).*(1-wy);
                w12 = (1-wx).*wy;
                w21 = wx.*(1-wy);
                w22 = wx.*wy;
                
                % Process all channels simultaneously using matrix operations
                img_reshaped = reshape(double(image), [], num_channels);
                warped_reshaped = reshape(warped, [], num_channels);
                
                % Vectorized interpolation for all channels
                interp_vals = w11 .* img_reshaped(i11,:) + ...
                             w12 .* img_reshaped(i12,:) + ...
                             w21 .* img_reshaped(i21,:) + ...
                             w22 .* img_reshaped(i22,:);
                
                warped_reshaped(valid_linear, :) = interp_vals;
                warped = reshape(warped_reshaped, out_height, out_width, num_channels);
            end
            
        case 'bicubic'
            % Floor coordinates
            x = floor(src_x);
            y = floor(src_y);
            dx = src_x - x;
            dy = src_y - y;
            
            % Find valid coordinates (need one extra pixel on each side for bicubic)
            valid = x >= 2 & x <= in_width-2 & y >= 2 & y <= in_height-2;
            valid_linear = find(valid);
            
            if ~isempty(valid_linear)
                % Extract valid coordinates
                x_valid = x(valid);
                y_valid = y(valid);
                dx_valid = dx(valid);
                dy_valid = dy(valid);
                
                % Pre-compute weights for x and y directions
                x_weights = zeros(length(valid_linear), 4, 'single');
                y_weights = zeros(length(valid_linear), 4, 'single');
                
                % Compute x weights vectorized
                for i = -1:2
                    x_weights(:,i+2) = bicubic_kernel(i - dx_valid);
                    y_weights(:,i+2) = bicubic_kernel(i - dy_valid);
                end
                
                % Create indices matrix for all 16 sample points
                indices = zeros(length(valid_linear), 16, 'single');
                idx = 1;
                for dy = -1:2
                    for dx = -1:2
                        indices(:,idx) = sub2ind([in_height, in_width], ...
                            y_valid+dy, x_valid+dx);
                        idx = idx + 1;
                    end
                end
                
                % Process all channels simultaneously
                img_reshaped = reshape(double(image), [], num_channels);
                warped_reshaped = reshape(warped, [], num_channels);
                
                % Get all sample points for all channels
                samples = img_reshaped(indices, :);
                samples = reshape(samples, [], 4, 4, num_channels);
                
                % Apply bicubic interpolation using matrix operations
                interp_vals = zeros(size(samples, 1), num_channels);
                for c = 1:num_channels
                    temp = squeeze(samples(:,:,:,c));
                    interp_vals(:,c) = sum(sum(y_weights .* reshape(temp, [], 4, 4) .* x_weights, 2), 3);
                end
                
                warped_reshaped(valid_linear, :) = uint8(max(0, min(255, interp_vals)));
                warped = reshape(warped_reshaped, out_height, out_width, num_channels);
            end
    end

function w = bicubic_kernel(x)
    % Vectorized bicubic kernel
    absx = abs(x);
    w = zeros(size(x), 'single');
    
    mask1 = absx <= 1;
    mask2 = absx <= 2 & ~mask1;
    
    absx2 = absx.^2;
    absx3 = absx.^3;
    
    w(mask1) = 1.5*absx3(mask1) - 2.5*absx2(mask1) + 1;
    w(mask2) = -0.5*absx3(mask2) + 2.5*absx2(mask2) - 4*absx(mask2) + 2;
end
end

function stitched = simpleBlendImages(warped_images, masks, options)
    % Blend warped images using masks with optional feathering
    % Args:
    %   warped_images: Cell array of warped images
    %   masks: Cell array of binary masks
    %   options: Struct with fields:
    %     - feather: boolean (enable/disable feathering)
    %     - kernel_size: size of Gaussian kernel (default: 31)
    %     - sigma: Gaussian sigma (default: kernel_size/6)
    %     - crop: boolean (enable/disable border cropping)
    %     - crop_padding: padding for cropping (default: 10)
    %
    % Example usage:   
    % Without feathering
    % options_no_feather = struct('feather', false, 'crop', true);
    % stitched = simpleBlendImages(warped_images, masks, options_no_feather);
    
    % % With feathering
    % options_feather = struct('feather', true, ...
    %                         'kernel_size', 31, ...
    %                         'sigma', 5, ...
    %                         'crop', true);
    % stitched = simpleBlendImages(warped_images, masks, options_feather);
    % 
    % % With custom parameters
    % options_custom = struct('feather', true, ...
    %                        'kernel_size', 51, ...
    %                        'sigma', 10, ...
    %                        'crop', true, ...
    %                        'crop_padding', 20);
    % stitched = simpleBlendImages(warped_images, masks, options_custom);

    % Default options
    if nargin < 3, options = struct(); end
    if ~isfield(options, 'feather'), options.feather = true; end
    if ~isfield(options, 'kernel_size'), options.kernel_size = 31; end
    if ~isfield(options, 'sigma'), options.sigma = options.kernel_size/6; end
    if ~isfield(options, 'crop'), options.crop = true; end
    if ~isfield(options, 'crop_padding'), options.crop_padding = 10; end
    
    % Get dimensions
    [output_h, output_w, ~] = size(warped_images{1});
    num_images = length(warped_images);
    
    % Convert all images to double for computation
    warped_double = cellfun(@double, warped_images, 'UniformOutput', false);
    
    if options.feather
        % Create Gaussian kernel
        kernel = fspecial('gaussian', ...
                         [options.kernel_size options.kernel_size], ...
                         options.sigma);
        
        % Apply feathering to all masks at once
        masks = cellfun(@(m) imfilter(m, kernel, 'replicate'), ...
                       masks, 'UniformOutput', false);
    end
    
    % Stack all masks into a 3D array for vectorized operations
    mask_stack = cat(3, masks{:});
    
    % Initialize accumulators
    accumulated = zeros(output_h, output_w, 3, 'double');
    weight_sum = sum(mask_stack, 3);
    
    % Blend all images at once
    for i = 1:num_images
        curr_mask = repmat(masks{i}, [1 1 3]);
        accumulated = accumulated + warped_double{i} .* curr_mask;
    end
    
    % Normalize
    weight_sum(weight_sum < eps) = 1;
    weight_sum_3d = repmat(weight_sum, [1 1 3]);
    stitched = uint8(accumulated ./ weight_sum_3d);
    
    % Crop black borders if requested
    if options.crop
        gray_img = rgb2gray(stitched);
        [rows, cols] = find(gray_img > 0);
        
        if ~isempty(rows) && ~isempty(cols)
            min_row = max(1, min(rows) - options.crop_padding);
            max_row = min(output_h, max(rows) + options.crop_padding);
            min_col = max(1, min(cols) - options.crop_padding);
            max_col = min(output_w, max(cols) + options.crop_padding);
            stitched = stitched(min_row:max_row, min_col:max_col, :);
        end
    end
end