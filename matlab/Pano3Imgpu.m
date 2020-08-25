function [ Image_d ] = Pano3Imgpu(Ipano, Params, r )
Ipano = imread('/media/chaoning/DiskFR/FocalDataprogressive/image_fun/pano_aamftnivhiesay.jpg');
[Ipano_H, Ipano_W, ~] = size(Ipano);
WW1 = gpuArray.linspace(1,Ipano_W, Ipano_W);
HH1 = gpuArray.linspace(1,Ipano_H, Ipano_H);
[WW, HH] = meshgrid(WW1, HH1);
for c=1:3
    %Ipano2one.Ipano = Ipano(:,:,c);
    Ipano1 = gpuArray(Ipano(:,:,c));
    Image_d(:,:,c) = interp2(WW, HH, im2double(Ipano1), WW, HH, 'nearest'); %; 
    %Image_d(:,:,c) = arrayfun(@interp2gpu, nx, ny);
end


end

