clear; clc; close all;
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   3DOF Serial Manipulator – Full Simulation & Analysis       ║\n');
fprintf('║   ED7220C-Style Robot (3DOF)  |  DH Convention               ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

%% Helper: DH to 4×4 homogeneous transform Standard DH homogeneous matrix
function T = dh2tform(a, d, alpha, theta)
    ct = cos(theta); st = sin(theta);
    ca = cos(alpha); sa = sin(alpha);
    T = [ct, -st*ca,  st*sa, a*ct;
         st,  ct*ca, -ct*sa, a*st;
          0,     sa,     ca,    d;
          0,      0,      0,    1];
end

function robot = buildRobotModel3DOF()
% BUILDROBOTMODEL3DOF  Define a 3-DOF ED7220C-style manipulator using
% Denavit-Hartenberg (DH) parameters and build a rigidBodyTree object.
%
% DH Convention (Standard Craig):
% T_{i-1,i} = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
%
% Robot configuration: Waist (J1) + Shoulder (J2) + Elbow (J3)
% This is a planar 3DOF arm operating in the vertical plane.
%
% Output struct fields:
%    robot.dh        – [3×4] DH table  [a, d, alpha, theta_offset]  (m / rad)
%    robot.qlim      – [3×2] joint limits (rad)
%    robot.rbt       – rigidBodyTree object
%    robot.L         – named link lengths (m)
%    robot.name      – string identifier

%% Link dimensions
L1 = 0.385;   % base-to-shoulder height  (d1)
L2 = 0.220;   % upper-arm length         (a2)
L3 = 0.220;   % forearm length           (a3)

robot.L    = struct('L1',L1,'L2',L2,'L3',L3);
robot.name = 'ED7220C_3DOF';

%% ── DH Parameter Table ───────────────────────────────────────────────────
%   Columns: [ a_i(m),  d_i(m),  alpha_i(rad),  theta_offset(rad) ]
%
%   Joint  | a      | d    | alpha   | theta_offset| Description
%   -------|--------|------|---------|-------------|------------------
%     1    |  0     |  L1  |  pi/2   |   0         | Waist  (base rotation)
%     2    |  L2    |  0   |  0      |   0         | Shoulder (vertical plane)
%     3    |  L3    |  0   |  0      |   0         | Elbow  (vertical plane)
%
%  Notes:
%   • Joint 1 (waist) rotates about Z with a pi/2 twist to lift into the
%     vertical plane – identical to the 6DOF base joint.
%   • Joints 2 & 3 are coplanar (alpha = 0), giving classic RRR kinematics.

robot.dh = [
     0,       L1,      pi/2,       0;   % joint 1 – waist
     L2,      0,       0,          0;   % joint 2 – shoulder
     L3,      0,       0,          0;   % joint 3 – elbow
];

%% Joint limits (radians)
robot.qlim = [
    -pi,      pi;       % joint 1  ±180°  (full waist rotation)
    -pi/2,    pi/2;     % joint 2  ±90°   (shoulder up/down)
    -pi*2/3,  pi*2/3;   % joint 3  ±120°  (elbow flex/extend)
];

%% Build rigidBodyTree
rbt = rigidBodyTree('DataFormat','column','MaxNumBodies',4);
rbt.Gravity = [0; 0; -9.81];

dhTable = robot.dh;
jnames  = {'Waist','Shoulder','Elbow'};

for i = 1:3
    body  = rigidBody(sprintf('link%d', i));
    joint = rigidBodyJoint(sprintf('joint%d', i), 'revolute');

    a     = dhTable(i,1);
    d     = dhTable(i,2);
    alpha = dhTable(i,3);
    theta = dhTable(i,4);

    % Build the fixed part of the DH transform
    Tfixed = dh2tform(a, d, alpha, theta);

    joint.HomePosition    = 0;
    joint.PositionLimits  = robot.qlim(i,:);
    setFixedTransform(joint, Tfixed);

    body.Joint = joint;

    if i == 1
        addBody(rbt, body, rbt.BaseName);
    else
        addBody(rbt, body, sprintf('link%d', i-1));
    end
end

% End-effector frame (fixed, no extra offset)
ee       = rigidBody('end_effector');
ee.Joint = rigidBodyJoint('ee_fixed','fixed');
setFixedTransform(ee.Joint, eye(4));
addBody(rbt, ee, 'link3');

robot.rbt = rbt;

%% Print summary table
fprintf('\n  ┌─────────────────────────────────────────────────────────┐\n');
fprintf('  │        DH PARAMETER TABLE  (SI units: m, rad)           │\n');
fprintf('  ├───────┬──────────┬──────────┬──────────┬────────────────┤\n');
fprintf('  │ Joint │  a (m)   │  d (m)   │alpha(rad)│ theta_offset   │\n');
fprintf('  ├───────┼──────────┼──────────┼──────────┼────────────────┤\n');
for i = 1:3
    fprintf('  │  %d %-8s│ %8.4f │ %8.4f │ %8.4f │ %8.4f       │\n', ...
        i, [jnames{i},' '], dhTable(i,1), dhTable(i,2), dhTable(i,3), dhTable(i,4));
end
fprintf('  └───────┴──────────┴──────────┴──────────┴────────────────┘\n\n');

end

fprintf('Building 3DOF DH model...\n');
robot = buildRobotModel3DOF();
