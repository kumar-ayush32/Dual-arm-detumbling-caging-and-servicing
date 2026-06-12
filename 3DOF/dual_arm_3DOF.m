L1 = 0.4;   % shoulder → elbow (in m)
L2 = 0.4;   % elbow    → wrist (in m)
L3 = 0.3;   % wrist    → EE (in m)
link_length = [L1, L2, L3];

figFK = figure('Name', 'Inverse Kinematics – 3DOF Planar 2D', ...
        'NumberTitle', 'off', 'Color', 'black', ...
        'Position', [100 100 700 650]);
ax = axes('Parent', figFK);
hold(ax, 'on');
axis(ax, 'equal');
grid(ax, 'on');

silent = true;
x11 = 0.08;  
y11 = 0.7;   
a31 = (1:step_size)/step_size*180;

x12 = -0.08; 
y12 = 0.7;   
a32 = (1:step_size)/step_size*180;

x01 = L1+L2+L3;   
y01 = 0;

x02 = -(L1+L2+L3); 
y02 = 0;

k = 1.2;
step_size = 100;
target_x = [0, x11, 0, x12]; target_y = [y11 - x11 , y11, y11 + x11 , y12];

x1 = linspace(x01, x11, step_size);
y_exp1 = y01 + (y11-y01)*(exp(k*(x1-x01))-1) ./ (exp(k*(x11-x01))-1);

x2 = linspace(x02, x12, step_size);
% y_exp2 = y02 + (y12-y02)*(exp(k*(x2-x02))-1) ./ (exp(k*(x12-x02))-1);

hold(ax,'on');

for i = 1:step_size
    [q_solutions1, valid1] = inverseKinematics3DOF(x1(i), y_exp1(i), a31(i), ...
        link_length, silent);
    [q_solutions2, valid2] = inverseKinematics3DOF(x2(i), y_exp1(i), a32(step_size+1-i), ...
        link_length, silent);

    cla(ax);
    plot2D(q_solutions1(1,:), link_length, ax, 'b');
    plot2D(q_solutions2(2,:), link_length, ax, 'g');
    fill(target_x, target_y, 'y')
    drawnow;

end




