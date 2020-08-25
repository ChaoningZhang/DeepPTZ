function Image_d = pano2Im(Ipano, Params, r)

%Parameters of the camera to generate
f = Params.f;
u0 = Params.W/2;  v0 = Params.H/2;
K = [f 0 u0; 0 f v0; 0 0 1];
xi = Params.xi; % distorsion parameters (spherical model)
[ImPano.H, ImPano.W, ~] = size(Ipano);

%tic
% 1. Projection on the image 
[grid_x, grid_y] = meshgrid(1:Params.W,1:Params.H);
X_Cam = grid_x./f - u0/f;
Y_Cam  = grid_y./f - v0/f;
Z_Cam =  ones(Params.H,Params.W);

%2. Image to sphere cart
% http://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
% Single View  Point  Omnidirectional Camera  Calibration from Planar Grids
% alpha_cam = ( xi.*Z_Cam + sqrt( Z_Cam.^2 + ...
%              ( (1-xi^2).*(X_Cam.^2 + Y_Cam.^2) ) ) ) ...
%              ./ (X_Cam.^2 + Y_Cam.^2 + Z_Cam.^2);      
% X_Sph = X_Cam.*alpha_cam;
% Y_Sph = Y_Cam.*alpha_cam;
% Z_Sph = (Z_Cam.*alpha_cam) - xi;
X_Sph = X_Cam;
Y_Sph = Y_Cam;
Z_Sph = Z_Cam;
%
% figure('name','spherical image');
% surf(X_Sph,Y_Sph,Z_Sph,'faceColor', 'texture','edgecolor', 'none','cdata',  double(Ipano)/255)
% axis equal;
% axis vis3d;

%3. Rotation of the sphere
x1 = r(1,1)*X_Sph + r(1,2)*Y_Sph + r(1,3)*Z_Sph;
y1 = r(2,1)*X_Sph + r(2,2)*Y_Sph + r(2,3)*Z_Sph;
z1 = r(3,1)*X_Sph + r(3,2)*Y_Sph + r(3,3)*Z_Sph;
X_Sph = x1; Y_Sph = y1; Z_Sph = z1;

%4. cart 2 sph
[ntheta, nphi, r] = cart2sph(X_Sph,Y_Sph, Z_Sph);

%5. Sphere to pano
min_theta = -pi; max_theta = pi;
min_phi = -pi/2; max_phi = pi/2;
min_x=1; max_x=ImPano.W;
min_y=1; max_y=ImPano.H;
% for x
a = (max_theta-min_theta)/(max_x-min_x);
b=max_theta-a*max_x; % from y=ax+b %% -a;
nx = (ntheta - b)/a;
% for y
a = (max_phi-min_phi)/(max_y-min_y);
b=max_phi-a*max_y; % from y=ax+b %% -a;
ny = (nphi - b)/a;

%6 Final step interpolation and mapping
Image_d=zeros(Params.H,Params.W,3);
for c=1:3
Image_d(:,:,c) = interp2(im2double(Ipano(:,:,c)), nx, ny, 'nearest');
end
%toc
%figure('name','output image'); imshow(Image_d)