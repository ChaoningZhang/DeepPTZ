function [rot_mat] = getRotationMat(roll,pitch,yaw)

%roll, pitch, yaw expressed in degree
% x, y, z

Rp=[cos(deg2rad(pitch)) 0 sin(deg2rad(pitch));...
    0 1 0;...
    -sin(deg2rad(pitch)) 0 cos(deg2rad(pitch))]; %Rot_y

Ry=[cos(deg2rad(yaw)) -sin(deg2rad(yaw)) 0;...
    sin(deg2rad(yaw)) cos(deg2rad(yaw)) 0;...
    0 0 1]; %Rot_z

Rr=[1 0 0;...
    0 cos(deg2rad(roll)) -sin(deg2rad(roll));...
    0 sin(deg2rad(roll)) cos(deg2rad(roll))]; %Rot_x

rot_mat = Ry*Rp*Rr;
% rot_mat = Rr*Rp*Ry;

end