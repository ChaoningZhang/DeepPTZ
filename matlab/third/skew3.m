function [V,dV] = skew3(v)
%SKEW3		[V,dV] = skew3(v)
%		Takes a 3 components vector and calculates
%		the corresponding skew-symmetric matrix.
%		It is useful for implementing the vector
%		product of 3-vectors: v x u = skew3(v) * u
%
%		dV (optional) returns the 9x3 matrix which represents
%		the 3x3x3 tensor of derivatives of V wrt v. 
%		

%	Updated 8/30/93

V = zeros(3,3);
V = [[0,-v(3),v(2)]; [v(3),0,-v(1)]; [-v(2),v(1),0]];

if (nargout >=2),
	dV = [0  0  0 ; 
	0  0 -1 ;
	0  1  0 ;
	0  0  1 ;
	0  0  0 ;
	-1 0  0 ;
	0 -1  0 ;
	1  0  0 ;
	0  0  0 ];
end;

return;

v = rand(3,1);	
eps = 1e-6;
for j=1:3,
	vp = v;
	vp(j) = v(j)+eps;
	dVtest(:,j) = qtoQ(1/eps*(skew3(vp) - skew3(v)));
end;

return;

% difference test
epsilon = 1e-6;
csi = randn(3,1);
rho = randn;
for (k = 1:3),
	csip = csi;
	csip(k) = csip(k)+epsilon;
	diff =  (skew3(csip)-skew3(csi))/epsilon;
	diff = diff';
	dFdcsi_test(:,k) = diff(:);
end;
[F,dFdcsi] = skew3(csi);
dFdcsi-dFdcsi_test,
norm(ans)