NYUv2_data = ''; % path to nyu_depth_v2_labeled.mat
split_file = ''; % path to splits.mat

test_root = '../Dataset/test';

if isempty(NYUv2_data) == 1
    ME = MException('Input:DataNotAssigned',...
        'you should assign the path of nyu_depth_v2_labeled.mat to the variable, NYUv2_data');
    throw(ME);
end
if isempty(split_file) == 1
    ME = MException('Input:DataNotAssigned',...
        'you should assign the path of splits.mat to the variable, split_file');
    throw(ME);
end

if ~exist(test_root, 'dir')
    mkdir(test_root);
end

targetSize = [240, 320];
% targetScale = (480-12)/240;
padding_1 = 6; % up and down
padding_2 = 8; % left and right

farPlane = 10; % largest depth value is 9.9955
nearPlane = 0.7; % smallest value is 0.7133 

NYUv2Data = load(NYUv2_data);
images = NYUv2Data.images;
depths = NYUv2Data.depths;
clear NYUv2Data

disp('loading data.. this may need a minute.');
splitIndx = load(split_file);
trainIndx = splitIndx.trainNdxs;
% testIndx = splitIndx.testNdxs;
trainNum = length(trainIndx);

[~,~,~,imageNum] = size(images);
if imageNum ~= 1449
    ME = MException('Input:DataNotAssigned',...
        'do not have 1449 images, check the nyu_depth_v2_labeled.mat.');
    throw(ME);
end

train_count = 0;
tic
for indx = 1:imageNum
    isTrain = false;
    if train_count+1 <= trainNum &&  trainIndx(train_count+1) == indx
        isTrain = true;
        train_count = train_count + 1;
    end
    
    if isTrain == false
        % resize image and convert depthdata
        RGBImage = images(:,:,:,indx);
        DepthMat = depths(:,:,indx);

        % crop white padding
        RGBImage = RGBImage(padding_1+1:end-padding_1,padding_2+1:end-padding_2,:);
        DepthMat = DepthMat(padding_1+1:end-padding_1,padding_2+1:end-padding_2);

        RGBImage = im2double(RGBImage);
        Depth = DepthMat;
        InfPos = find(Depth > farPlane);
        Depth(InfPos) = farPlane;
        Depth = single(Depth);
        zerosPos = find(Depth <= 0);
        Depth(zerosPos) = (rand(1)+1);

        % scale to target size
        RGBImage = imresize(RGBImage, [480, 640]);
        Depth = imresize(Depth, [480, 640]);

        RGBImage_rs = imresize(RGBImage, targetSize);
        Depth_rs = imresize(Depth, targetSize);
        Depth_rsx2 = single(imresize(Depth_rs, 1/2));
        Depth_rsx4 = single(imresize(Depth_rs, 1/4));
        Depth_rsx8 = single(imresize(Depth_rs, 1/8));

        Depth_rs_t = log(Depth_rs); 
        Depth_rsx2_t = log(Depth_rsx2);
        Depth_rsx4_t = log(Depth_rsx4);
        Depth_rsx8_t = log(Depth_rsx8);

        data.rgb = RGBImage_rs;
        data.depth = Depth_rs_t;
        data.depthx2 = Depth_rsx2_t;
        data.depthx4 = Depth_rsx4_t;
        data.depthx8 = Depth_rsx8_t;
        data.realDepth = Depth_rs;
        data.imageSize = size(Depth_rs);

        saveFile = [test_root, '/nyu_v2_', num2str(indx), '.mat'];
        save(saveFile, 'data');
    end
    
    if mod(indx, 10) == 0
        disp([num2str(indx),' images has been processed!']);
        toc
    end
end

disp([num2str(imageNum),' images has been processed!']);



