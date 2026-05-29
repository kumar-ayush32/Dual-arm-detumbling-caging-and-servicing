function T = dh_transform(alpha, a, d, theta)
% DH_TRANSFORM - Builds a single 4×4 D-H homogeneous transformation matrix.
%
%   Standard D-H convention:
%       T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
%
%   This expands to the closed-form matrix below.
%
%   Inputs:
%       alpha  - Twist angle about x_{i-1} axis (rad)
%       a      - Link length along x_{i-1} axis (mm)
%       d      - Link offset along z_i axis (mm)
%       theta  - Joint angle about z_i axis (rad)
%
%   Output:
%       T      - 4×4 homogeneous transformation matrix

    ct = cos(theta);
    st = sin(theta);
    ca = cos(alpha);
    sa = sin(alpha);

    T = [ct,   -st*ca,   st*sa,   a*ct;
         st,    ct*ca,  -ct*sa,   a*st;
          0,       sa,      ca,      d;
          0,        0,       0,      1];

end