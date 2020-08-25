function [ x1, y1, z1 ] = rotation( X_Sph, Y_Sph, Z_Sph, r11, r12, r13, r21, r22, r23, r31, r32, r33 )
%ROTATION Summary of this function goes here
%   Detailed explanation goes here
x1 = r11*X_Sph + r12*Y_Sph + r13*Z_Sph;
y1 = r21*X_Sph + r22*Y_Sph + r23*Z_Sph;
z1 = r31*X_Sph + r32*Y_Sph + r33*Z_Sph;

end

