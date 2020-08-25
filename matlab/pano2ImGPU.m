function Image_d = pano2ImGPU(Ipano, Params, r)

%Parameters of the camera to generate
f = Params.f;
u0 = Params.W/2;  v0 = Params.H/2;
K = [f 0 u0; 0 f v0; 0 0 1];
xi = Params.xi; % distorsion parameters (spherical model)
[ImPano.H, ImPano.W, ~] = size(Ipano);

%tic
% 1. Projection on the image 
xx = gpuArray.linspace(1,Params.W, Params.W);
yy = gpuArray.linspace(1,Params.H, Params.H);
[grid_x, grid_y] = meshgrid(xx,yy);
X_Cam = grid_x./f - u0/f;
Y_Cam  = grid_y./f - v0/f;
Z_Cam =  ones(Params.H,Params.W);

%2. Image to sphere cart
% http://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
% Single View  Point  Omnidirectional Camera  Calibration from Planar Grids

%Error using gpuArray/sqrt
%SQRT: needs to return a complex result, but this is not supported for real input X on the GPU. Use
%SQRT(COMPLEX(X)) instead.
alpha_cam = ( xi.*Z_Cam + real(sqrt(complex(Z_Cam.^2 + ...
             ( (1-xi^2).*(X_Cam.^2 + Y_Cam.^2)) ) )) ) ...
             ./ (X_Cam.^2 + Y_Cam.^2 + Z_Cam.^2);      
X_Sph = X_Cam.*alpha_cam;
Y_Sph = Y_Cam.*alpha_cam;
Z_Sph = (Z_Cam.*alpha_cam) - xi;
%X_Sph = X_Cam;
%Y_Sph = Y_Cam;
%Z_Sph = Z_Cam;
%
% figure('name','spherical image');
% surf(X_Sph,Y_Sph,Z_Sph,'faceColor', 'texture','edgecolor', 'none','cdata',  double(Ipano)/255)
% axis equal;
% axis vis3d;

%3. Rotation of the sphere
[ x1, y1, z1 ] = arrayfun(@rotation, X_Sph, Y_Sph, Z_Sph, r(1,1), r(1,2), r(1,3), r(2,1), r(2,2), r(2,3), r(3,1), r(3,2), r(3,3));
X_Sph = x1; Y_Sph = y1; Z_Sph = z1;

%4. cart 2 sph
[ntheta, nphi, r] = arrayfun(@cart2sph, X_Sph,Y_Sph, Z_Sph);

%5. Sphere to pano
min_theta = -pi; max_theta = pi;
min_phi = -pi/2; max_phi = pi/2;
min_x=1; max_x=ImPano.W;
min_y=1; max_y=ImPano.H;
% for x
a_theta = (max_theta-min_theta)/(max_x-min_x);
b_theta=max_theta-a_theta*max_x; % from y=ax+b %% -a;
%nx = (ntheta - b_theta)/a_theta;
% for y
a_phi = (max_phi-min_phi)/(max_y-min_y);
b_phi=max_phi-a_theta*max_y; % from y=ax+b %% -a;
%ny = (nphi - b_phi)/a_phi;
[ nx, ny] = arrayfun(@angle2xy, ntheta, a_theta, b_theta, nphi, a_phi, b_phi);
%6 Final step interpolation and mapping
%Image_d=zeros(Params.H,Params.W,3);
%global Ipano_global
%Ipano_global = Ipano;
nx = gather(nx);
ny = gather(ny);

% Ipano = imread('/media/chaoning/DiskFR/FocalDataprogressive/image_fun/pano_aamftnivhiesay.jpg');
% [Ipano_H, Ipano_W, ~] = size(Ipano);
% WW1 = gpuArray.linspace(1,Ipano_W, Ipano_W);
% HH1 = gpuArray.linspace(1,Ipano_H, Ipano_H);
% [WW, HH] = meshgrid(WW1, HH1);
for c=1:3
    %Ipano2one.Ipano = Ipano(:,:,c);
    %Ipano1 = gpuArray(Ipano(:,:,c));
    Image_d(:,:,c) = interp2(im2double(gpuArray(Ipano(:,:,c))), nx, ny, 'nearest'); %; 
    %Image_d(:,:,c) = arrayfun(@interp2gpu, nx, ny);
end
end
function [ output ] = interp2gpu(nx, ny)
%INTERP2GPU Summary of this function goes here
%   Detailed explanation goes here
output = nx +ny;
%output = interp2(im2double(Ipano), nx, ny, 'nearest');

end
%toc
% figure('name','output image'); imshow(Image_d)