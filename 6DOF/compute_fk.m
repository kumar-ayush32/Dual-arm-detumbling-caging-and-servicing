function [T_final, positions] = compute_fk(robot, q_deg)
    % Forward kinematics for 6DOF robot.
    q_rad = deg2rad(q_deg);
    positions = zeros(3, robot.dof + 1);
    positions(:,1) = [0;0;0];
    T_cumulative = eye(4);

    %% FK Loop
    for i = 1:robot.dof

        alpha_i = robot.alpha(i);
        a_i     = robot.a(i);
        d_i     = robot.d(i);
        theta_i = q_rad(i);

        T_i = dh_transform(alpha_i, a_i, d_i, theta_i);
        T_cumulative = T_cumulative * T_i;
        positions(:,i+1) = T_cumulative(1:3,4);

    end
    T_final = T_cumulative;

end