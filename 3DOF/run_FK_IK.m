q = [45, 60, 30];
T_final = forwardKinematics3DOF(q);
x = T_final(1,4);
y = T_final(2,4);
a3 = atan2d(T_final(2,1), T_final(1,1));

L1 = 0.3;   % shoulder → elbow (in m)
L2 = 0.2;   % elbow    → wrist (in m)
L3 = 0.2;   % wrist    → EE (in m)
link_length = [L1, L2, L3];
silent = false;
[q_solutions, valid] = inverseKinematics3DOF(x, y, a3, link_length, silent);