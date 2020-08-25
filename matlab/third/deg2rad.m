function R=deg2rad(D)

%DEG2RAD Converts angles from degrees to radians
%
%  rad = DEG2RAD(deg) converts angles from degrees to radians.
%
%  See also RAD2DEG, DEG2DMS, ANGLEDIM, ANGL2STR

%  Copyright 1996-1998 by Systems Planning and Analysis, Inc. and The MathWorks, Inc.
%  Written by:  E. Byrns, E. Brown
%   $Revision: 1.7 $    $Date: 1998/08/10 17:47:35 $


if nargin==0
	error('Incorrect number of arguments')
elseif ~isreal(D)
     warning('Imaginary parts of complex ANGLE argument ignored')
     D = real(D);
end

R = D*pi/180;