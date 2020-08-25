function	[out,dout]=rodrigues(in)
%
% RODRIGUES		Transform rotation matrix into rotation vector.
%		
%			Syntax:  [out,d_out_d_in]=RODRIGUES(in)
% 			If IN is a 3x3 rotation matrix then OUT is the
%			corresponding 3x1 rotation vector
%
% 			if IN is a rotation 3-vector then OUT is the 
%			corresponding 3x3 rotation matrix
%
%	d_out_d_in is the derivative of the output wrt the input
%
%
%			Makes use of the following subroutines:
%
%				-> ~/src/matlab/detensor.m
%				-> ~/src/matlab/dABdA.m
%				-> ~/src/matlab/dABdB.m

%	Written by Andrea Mennucci  7/28/93 
%	Modified by Stefano Soatto
%	Copyright (c) California Institute of Technology

[m,n] = size(in);
bigeps = (10e+5)*eps;

if ((m==1) & (n==3)) | ((m==3) & (n==1))	%% it is a rotation vector
	theta = norm(in);
	if n==length(in)  in=in'; end; %% make it a column vec. if necess.
	omega = in;
	if theta < sqrt(eps)*1e2,
	   theta2 = in' * in ;
	   sinthetatheta =1-theta2/6;
	   onecosthetatheta2=1-theta2/24;
	   
	   if nargout>1,
	    dsinthetatheta_domega=- omega'/3;
	    donecosthetatheta2_domega= -omega'/12;
	   end;
	else;
	   theta2 = theta * theta ;
	   onecosthetatheta2=(1-cos(theta))/theta2;
	   sinthetatheta=sin(theta)/theta;
	   
	   if nargout>1,
	    dtheta_domega=omega'/theta;
	    dsinthetatheta_dt=(cos(theta)*theta - sin(theta))/theta2;
	    donecosthetatheta2_dt=  ...
		(sin(theta)*theta2-(1-cos(theta))*theta*2)/theta2/theta2;
	    dsinthetatheta_domega=dsinthetatheta_dt * dtheta_domega;
	    donecosthetatheta2_domega=donecosthetatheta2_dt * dtheta_domega;
	   end;
	end;
	omegav=skew3(omega);
	R = eye(3) + omegav*sinthetatheta + omegav*omegav*onecosthetatheta2 ;
	if nargout>1,
	   domegav_domega=[[0 0 0];[0 0 -1];[0 1 0];[0 0 1];[0 0 0]; ...
		[-1 0 0];[0 -1 0];[1 0 0];[0 0 0]];
	   domegav2_domegav=dABdA(omegav,omegav)+dABdB(omegav,omegav);
	   dRdomega=domegav_domega*sinthetatheta + ...
		detensor(omegav)* dsinthetatheta_domega + ...
		domegav2_domegav*domegav_domega* onecosthetatheta2 + ...
		detensor(omegav*omegav)*donecosthetatheta2_domega;
	   dout=dRdomega;
	end;
	out = R;

 %% it is a rot matrix
elseif ((m==n) & (m==3)),
	if (norm(in' * in - eye(3),Inf) >bigeps) | (abs(det(in)-1) > bigeps) ,
	 disp('( rodrigues: matrix is not a rotation matrix)'); end;
	R = in;
        trc=trace(R);
        trc2=(trc-1)/2;
	sinacostrc2=sqrt(1- trc2*trc2);

	s=[R(3,2)-R(2,3), R(1,3)-R(3,1), R(2,1)-R(1,2)]';
	if (1- trc2*trc2) >= eps,
	  tHeta = (acos(trc2));
	  tHetaf=tHeta/ (2 *sin(tHeta));
	  dtHetaf_dt=(2*sin(tHeta)-tHeta*2*cos(tHeta)) ...
		/ (2 *sin(tHeta))/(2 *sin(tHeta));
	  if nargout>1,
	    dtrcdR=[ 1 0 0 0 1 0 0 0 1];
	    
	    dtHetadtrc=-0.5 / sinacostrc2 ;
	    dtHetadR = dtHetadtrc * dtrcdR;
	  end;
	else;
	 tHeta = real(acos(trc2));
	 tHetaf=0.5/(1-tHeta/6); % = ~ 1/2+tHeta/12
	 dtHetaf_dt=1/12;
	 dtHetadR=zeros(1,9);% this is infinity 
	 disp('approximate result in rodrigues');
	end;
	omega = tHetaf * s;
	out=omega;

	if nargout>1,
	  dsdR=[ [ 0 0 0  0 0 -1  0 1 0 ];[ 0 0 1  0 0 0  -1 0 0 ]; ...
		  [ 0 -1 0  1 0 0  0 0 0 ]] ;
	  
	  domegadR = tHetaf * dsdR + s * dtHetaf_dt * dtHetadR  ;
	  dout=domegadR;
	end;
else
	error('Rodrigues: Neither a matrix nor a rotation vector were provided');
end;

return; 

% differece tests

Rtest = eye(3)+ skew3(csi)/norm(csi)*sin(norm(csi)) + skew3(csi)^2/(norm(csi)^2)*(1-cos(norm(csi)))


% Test for rotation vectors -> rotation matrices
for i = 1: 10, 
	epsilon = 10^(-i);
	ddelta(i) = epsilon;
	csi = .01*randn(3,1);
	for (k = 1:3),
		csip = csi;
		csip(k) = csip(k)+epsilon;
		diff =  (rodrigues(csip)-rodrigues(csi))/epsilon;
		diff = diff';
		dFdcsi_test(:,k) = diff(:);
	end;
	[F,dFdcsi] = rodrigues(csi);
	nnorm(i)=norm(dFdcsi-dFdcsi_test);
end;
plot(log10(ddelta),log10(nnorm));
% Test for rotation matrices -> vectors
epsilon = 1e-6;
csi = randn(3,1);
csirod = dordigues(csi);
csirod = csirod';
csirod = csirod(:);
for (k = 1:9),
	csirodp = csirod;
	csirodp(k) = csirodp(k)+epsilon;
	diff =  (rodrigues(csip)-rodrigues(csi))/epsilon;
	diff = diff'
	dFdcsi_test(:,k) = diff(:);
end;
[F,dFdcsi] = rodrigues(csi);
dFdcsi-dFdcsi_test,
norm(ans)


%%%%%%%%% test 
o=[ 0.3 0.1 0.2]';
o=rand(3,1)*pi;
o=rand(3,1)*1e-5;
minidelta=1e-7;
R=rodrigues(o);

[temp,OORR]=rodrigues(R); 
%temp'-o'
o=temp;
for i=1:3,
 for j=1:3,
   R2=R;
   R2(i,j)=R2(i,j)+ minidelta;  
   [o2]=rodrigues(R2);
   OORR2(:,(i-1)*3+j)=(o2-o)/minidelta;          
 end;
end;
(OORR2 -OORR)/pi
max(max(OORR))/pi
%%%%%%%%% test 2
o=[ 0.3 0.1 0.2]';
o=rand(3,1)/pi;
minidelta=1e-6;
o=[ 0. 0. pi/4]';
o=rand(3,1)*pi;
o=rand(3,1)*1e-8;
o=rand(3,1)*1e-3;

[R,RROO]=rodrigues(o);
for i=1:3,
   o2=o;
   o2(i)=o2(i)+ minidelta;  
   [R2]=rodrigues(o2);
   RROO2(:,i)=detensor(R2-R)/minidelta;          
end;
(RROO2 -RROO)
max(max(RROO))

for j=1:30
 o=rand(3,1)*pi*2;
 if norm(o-rodrigues(rodrigues(o)))>1e-19, error(''); end;
end;