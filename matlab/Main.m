clear all; close all ; clc;
gpuDevice(2)
debug = false;
addpath('third')

% Prepare the file names of all the images
[ FList ] = ReadFileNames('/media/user/SSD1TB-0/camera/indoor');
%[ FList ] = ReadFileNames('image_fun');
%% Split
TrainPer = 0.9;
ValidationPer = 0.05;
TestPer = 0.05;
TotalPano = length(FList)
rand('seed',1);
[trainInd,valInd,testInd] = dividerand(TotalPano,TrainPer,ValidationPer,TestPer);
fprintf('Before dealing with pano corrupton:\n')
fprintf('%d panos for train\n %d panos for val\n %d panos for test\n', length(trainInd), length(valInd), length(testInd));
%% Set initial values for parameters
Params.W = 299;
Params.H = 299;
%Params.f = 250;
Params.xi = 0;
Rot.x = 0; Rot.y=0; Rot.z = 0;
RotC.x = 0; RotC.y=0; RotC.z = 0;
ImagePerPano = 10;

%% dataset table parameters
focal_avg = 275%150;
focal_var = 50;
%focal_range = 0;
%degree_range = 0;
lap_Ind = 1;
lap_list=[];


%% to deal with corrupted images
error_train_list = {};
error_val_list = {};
error_test_list = {};
error_train = 0;
error_val = 0;
error_test = 0;

%% folders

folder_all = 'All_images/'
% mkdir(strcat(folder_all, 'Train/Image1'));
% mkdir(strcat(folder_all, 'Train/Image2'));
% mkdir(strcat(folder_all, 'Val/Image1'));
% mkdir(strcat(folder_all, 'Val/Image2'));
% mkdir(strcat(folder_all, 'Test/Image1'));
% mkdir(strcat(folder_all, 'Test/Image2'));

%% Generation of Train/Val/Test datasets
tic
imagechoice = 0;
for focal_range = [225]%100]
    for degree_range = [15]
        focal_degree = strcat('focal', int2str(focal_range), 'degree', int2str(degree_range), '/')
        make_directory(focal_degree, folder_all);
        for set_name = ["Train", "Test", "Val"]
            set_name = char(set_name)
            csv_file = fopen(strcat('All_images/', focal_degree, set_name, '/', set_name, '.csv'),'w');
            fprintf(csv_file,'Im1,Im2, roll1, pitch1, yaw1, focal1, focal2, roll2, pitch2, yaw2, distor1, distor2, rollself, pitchself, yaw\n'); % write to the csv file

            if strcmp(set_name, 'Train')
                ListInd = trainInd;
                %ListRandom = randi(length(ListInd),1800,1);
            elseif strcmp(set_name, 'Val')
                ListInd = valInd;
                %ListRandom = randi(length(ListInd),200,1);
            elseif strcmp(set_name, 'Test')
                ListInd = testInd;
                %ListRandom = randi(length(ListInd),200,1);
            end
            
            ImageIndex = 0;            
            for i = 1 : length(ListInd)
                warning('');
                filename = FList{ListInd(i)};

                %%%%%% to deal with corrupted images %%%%%%%
%                 tic

                [Ipano, abnormal, error_train, error_val, error_test] = image_read(filename, lastwarn, error_train, error_val, error_test);
                if abnormal
                    continue
                end
%                 toc
                %%%%%% to deal with corrupted images %%%%%%

                fprintf(strcat(focal_degree, set_name, ': Pano', int2str(i), '\n'));

                ImagesLeft = ImagePerPano;
                %trymax = 50;
                while ImagesLeft > 0
                    ImagesLeft = ImagesLeft -1;
                    Rot.x = ((rand-0.5)*2)*15; % roll up
                    Rot.y = ((rand-0.5)*2)*180; %pitch right
                    Rot.z = ((rand-0.5)*2)*15; %yaw clockwise
                    r = getRotationMat(90,0,90)*getRotationMat(Rot.x,Rot.y,Rot.z)';

                    RotC.x = ((rand-0.5)*2)*degree_range; % roll 
                    RotC.y = ((rand-0.5)*2)*degree_range; %pitch
                    RotC.z = ((rand-0.5)*2)*degree_range; %yaw 
                    rc = getRotationMat(RotC.x,RotC.y,RotC.z);
                    RotC_inv = rotm2eul(rc')*180/pi;
                    RotC.xx = RotC_inv(3,1);
                    RotC.yy = RotC_inv(2,1);
                    RotC.zz = RotC_inv(1,1);
                    

                    %imagechoice = imagechoice + 1;
                    %if mod(imagechoice,2) == 0
                    Params.f1 = focal_avg + (rand-0.5)*2*focal_range; % divide by 2 Params.f1 = focal_avg + (rand-0.5)*2*focal_range;
                    Params.f2 = Params.f1 + (rand-0.5)*2*focal_var;
                    Params.f2 = min(max(Params.f2, 50), 500);
                    Params.xi1 = 0.5 + (rand-0.5)*2*0.5;
                    Params.xi2 = Params.xi1 - 0.1*rand*sign(Params.f2 - Params.f1); % large f means small xi
                    Params.xi2 = min(max(Params.xi2, 0), 1);
                    %end
%                     if mod(imagechoice,2) == 1
%                         Params.f2 = focal_avg + (rand-0.5)*2*(focal_range-focal_var); % divide by 2 Params.f1 = focal_avg + (rand-0.5)*2*focal_range;
%                         Params.f1 = Params.f2 + (rand-0.5)*2*focal_var;  
%                         Params.xi2 = 0.5 + (rand-0.5)*2*(0.5-0.1);
%                         Params.xi1 = Params.xi2 - 0.1*rand*sign(Params.f1 - Params.f2);
%                     end
                    %Params.f = focal_avg + (rand-0.5)*2*focal_range; %randi([50 300])
                    %Params.x = ((rand-0.5)*2)*5; Params.y = ((rand-0.5)*2)*5;
%                     K1 = [Params.f 0 Params.W/2; 0 Params.f Params.H/2; 0 0 1];
%                     K2 = K1;
%                     H = (K2)*rc*inv(K1); 
%                     H=inv(H); H=H./H(3,3); % from end to start, H(3,3) to be compatible with the spatial_interp
%                     %Compute intersection
%                     b1 = ones(Params.W, Params.H);
%                     b2 = spatial_interp(b1, (H), 'linear', 'homography', 1:Params.W, 1:Params.H);
%                     b2(b2>0)=1; % to convert float to integer
%                     overlap = b1 & b2;
%                     AreaPerc = sum(sum(overlap))/(Params.W*Params.H);
% 
%                     if (AreaPerc<0.4)
%                         fprintf('AreaPerc: %f, rejected\n', AreaPerc);
%                         continue%need to rejected
%                     end
%                     tic
%                     Image_d1 = pano2Im(Ipano, Params, r);
%                     Image_d2 = pano2Im(Ipano, Params, r*rc');
%                     toc
%                     tic
                    %Params.xi = rand*1.2;
                    Params.f = Params.f1;
                    Params.xi = Params.xi1;
                    Image_d1 = pano2ImGPU(Ipano, Params, r);
                    Params.f = Params.f2;
                    Params.xi = Params.xi2;
                    Image_d2 = pano2ImGPU(Ipano, Params, r*rc');
                    Image_d1 = gather(Image_d1);
                    Image_d2 = gather(Image_d2);
%                     toc
                    %imshow([Image_d1 Image_d2 Image_d3 Image_d4])

                    %lap = fspecial('laplacian');
%                     fprintf('lap\n')
%                     tic
                    im_lap   = imfilter(Image_d1(:,:,1), fspecial('laplacian'), 'replicate', 'conv');
                    lap_sum = sum(sum(abs(im_lap)));
%                     if lap_sum < 3500
%                         fprintf('lap_sum: %s image no feature, rejected\n', lap_sum)
%                         continue
%                     end
                    lap_list(lap_Ind) = lap_sum;
                    lap_Ind = lap_Ind +1;
%                     toc
                    
%                     ImageIndex
%                     Params.f
%                     gradient = sum(sum(abs(im_lap)))
%                     gradient = sum(sum(abs(im_lap).*overlap))


%                     [b3, flow] = spatial_interp_and_flow(Image_d1(:,:,1), (H), 'linear', 'homography', 1:Params.W, 1:Params.H);
%                     overlap3(:,:,1) = overlap; overlap3(:,:,2) = overlap; overlap3(:,:,3) = overlap;
%                     imshow([Image_d1 Image_d2 overlap3] ); %drawnow
                    name1 = strcat(folder_all, focal_degree, set_name, '/Image1/Im', int2str(ImageIndex), '.png'); %['All_images/Val/Image1/Im', int2str(ImageIndex), '.png'];
                    name2 = strcat(folder_all, focal_degree, set_name, '/Image2/Im', int2str(ImageIndex), '.png');
%                     tic
                    imwrite(Image_d1, name1); % generate image1
                    imwrite(Image_d2, name2); % generate image2
%                     toc
                    %fprintf(csv_file,'%s, %s, %f, %f, %f, %f, %f, %f\n',name1, name2, double(RotC.x), double(RotC.y), double(RotC.z), double(Params.f), double(Params.x), double(Params.y)); % write to the csv file
                    fprintf(csv_file,'%s,%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n',name1, name2, double(RotC.x), double(RotC.y), double(RotC.z), double(Params.f1), double(Params.f2), double(RotC.xx), double(RotC.yy), double(RotC.zz), double(Params.xi1), double(Params.xi2), double(Rot.x), double(Rot.y), double(Rot.z)); % write to the csv file
                    ImageIndex = ImageIndex+1;

                end

                if debug
                    if i == 1
                       break
                    end
                end
            end
            fclose(csv_file);
        end
        
    end
end
toc
fprintf('After dealing with pano corrupton:\n')
fprintf('%d panos for train\n %d panos for val\n %d panos for test\n', length(trainInd)-error_train, length(valInd)-error_val, length(testInd)-error_test);

function [Ipano, abnormal, error_train, error_val, error_test] = image_read(filename, lastwarn, error_train, error_val, error_test)
    abnormal = false;
    try
        Ipano = imread(filename);
        [warnMsg, ~] = lastwarn;
        if ~isempty(warnMsg)
            if strcmp(set_name, 'Train')
                error_train = error_train + 1;
            elseif strcmp(set_name, 'Val')
                error_val = error_val +1;
            elseif strcmp(set_name, 'Test')
                error_test = error_test +1;
            end
            fprintf(warnMsg); fprintf('\n');
            abnormal = true;
        end
    catch ME
        if strcmp(ME.message, 'Unable to determine the file format.')
            if strcmp(set_name, 'Train')
                error_train = error_train + 1;
            elseif strcmp(set_name, 'Val')
                error_val = error_val +1;
            elseif strcmp(set_name, 'Test')
                error_test = error_test +1;
            end
            fprintf(ME.message); fprintf('\n');
            abnormal = true; 
        end
    end
end

function make_directory(folder_all, focal_degree)
    mkdir(strcat(focal_degree, folder_all, 'Train/Image1'));
    mkdir(strcat(focal_degree, folder_all, 'Train/Image2'));
    mkdir(strcat(focal_degree, folder_all, 'Val/Image1'));
    mkdir(strcat(focal_degree, folder_all, 'Val/Image2'));
    mkdir(strcat(focal_degree, folder_all, 'Test/Image1'));
    mkdir(strcat(focal_degree, folder_all, 'Test/Image2'));
end
