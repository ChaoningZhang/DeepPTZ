function [ nx, ny] = angle2xy( ntheta, a_theta, b_theta, nphi, a_phi, b_phi )
%ANGLE2XY Summary of this function goes here
%   Detailed explanation goes here
nx = (ntheta - b_theta)/a_theta;
ny = (nphi - b_phi)/a_phi;
end

