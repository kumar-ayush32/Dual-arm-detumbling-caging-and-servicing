fig = figure('Name', '6DOF Robot', 'NumberTitle', 'off', ...
    'Color', 'black', 'Position', [100 100 700 650]);
ax = axes('Parent', fig);
hold(ax, 'on');
axis(ax, 'equal');
grid(ax, 'on');
view(-15,15)
xlabel(ax, 'X');
ylabel(ax, 'Y');
zlabel(ax, 'Z');
xlim(ax,[-300,300]);
ylim(ax,[-100,100]);
zlim(ax,[-50,200]);

% Initial Guess for IK
target_pos = [-13.79; -49.50; 150];

base_pos1 = [30;0;0];
target_centre1 = target_pos - base_pos1;
base_pos2 = [-30;0;0];
target_centre2 = target_pos - base_pos2;
s = 10;
target_pos1 = target_centre1 + [sqrt(3)*s/2;sqrt(3)*s/2;sqrt(3)*s/2];
q01 = [0,0,45,45,45,0];
target_pos2 = target_centre2 + [-sqrt(3)*s/2;-sqrt(3)*s/2;-sqrt(3)*s/2];
q02 = [-123,56,1,89,-45,60];

step_size = 100;

xc = target_pos(1);
yc = target_pos(2);
zc = target_pos(3);

% Cube vertices
V1 = [xc-s yc-s zc-s; xc+s yc-s zc-s; xc+s yc+s zc-s; xc-s yc+s zc-s;
    xc-s yc-s zc+s; xc+s yc-s zc+s; xc+s yc+s zc+s; xc-s yc+s zc+s];
F1 = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
p1 = base_pos1(:)';
p2 = base_pos2(:)';
w = 6;
v = p2 - p1; v = v / norm(v);
if abs(dot(v,[0 0 1])) < 0.9
    temp = [0 0 1];
else
    temp = [0 1 0];
end
u1 = cross(v,temp); u1 = u1 / norm(u1);
u2 = cross(v,u1); u2 = u2 / norm(u2);
u1 = w*u1; u2 = w*u2;
V2 = [p1 + u1 + u2; p1 + u1 - u2; p1 - u1 - u2; p1 - u1 + u2;
    p2 + u1 + u2; p2 + u1 - u2; p2 - u1 - u2; p2 - u1 + u2;];
F2 = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];

hold(ax,'on');
robot = model_6DOF();
silent = false;

initial_pos1 = [robot.L.L1+robot.L.L2+robot.L.L3+robot.L.L4+robot.L.L5+robot.L.L6-30; 0; 0];
initial_pos2 = [-(robot.L.L1+robot.L.L2+robot.L.L3+robot.L.L4+robot.L.L5+robot.L.L6-30); 0; 0];
r1 = norm(initial_pos1); r2 = norm(initial_pos2);
r3 = norm(target_pos1); r4 = norm(target_pos2);

P1 = initial_pos1 / r1;
P2 = initial_pos2 / r2;
P3 = target_pos1  / r1;
P4 = target_pos2  / r2;
theta1 = acos(max(-1,min(1,dot(P1,P3))));
theta2 = acos(max(-1,min(1,dot(P2,P4))));
t = linspace(0,1,step_size);

for i = 1:step_size
    %Solve IK
    a = sin((1-t(i))*theta1) / sin(theta1);
    b = sin(t(i)*theta1) / sin(theta1);
    c = sin((1-t(i))*theta2) / sin(theta2);
    d = sin(t(i)*theta2) / sin(theta2);
    traj1 = r1 * (a*P1 + b*P3);
    traj2 = r2 * (c*P2 + d*P4);
    q_sol1 = inverseKinematics6DOF(robot, traj1, q01);
    q_sol2 = inverseKinematics6DOF(robot, traj2, q02);
    
    cla(ax);
    robot_plot(robot, q_sol1, silent, "INVERSE KINEMATICS", ax, base_pos1);
    robot_plot(robot, q_sol2, silent, "INVERSE KINEMATICS", ax, base_pos2);
    
    patch('Parent', ax, 'Vertices', V1, 'Faces', F1, ...
        'FaceColor', 'y', 'FaceAlpha', 0.3, ...
        'EdgeColor', 'y', 'LineWidth', 2);
    patch('Parent', ax, 'Vertices', V2, 'Faces', F2, 'FaceColor', [0.2 0.8 1], ...
        'FaceAlpha', 1, 'EdgeColor', 'c', 'LineWidth', 2);
    drawnow;
end

