function robot = model_6DOF()
    % Link lengths (in mm)
    robot.L.L1 = 6;
    robot.L.L2 = 80;
    robot.L.L3 = 70;
    robot.L.L4 = 50;
    robot.L.L5 = 40;
    robot.L.L6 = 30;
    robot.dof = 6;

    % Standard DH Parameters
    robot.alpha = [pi/2, 0, 0, pi/2, -pi/2, 0];
    robot.a = [0, robot.L.L2, robot.L.L3, robot.L.L4, robot.L.L5, robot.L.L6 ];
    robot.d = [robot.L.L1, 0, 0, 0, 0, 0];

end