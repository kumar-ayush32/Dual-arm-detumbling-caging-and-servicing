function robot_plot(robot, q_sol, silent, fig_title, ax, base_pos)
    [T_final, all_positions] = compute_fk(robot, q_sol);
    end_effector_pos = T_final(1:3,4) + base_pos;

    all_positions = all_positions + base_pos;
    x = all_positions(1,:);
    y = all_positions(2,:);
    z = all_positions(3,:);

    % Plot Links
    plot3(ax, x, y, z, '-o', 'Color', [0 0.4470 0.7410], ...
        'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r');

    %% End-Effector Claw
    R = T_final(1:3,1:3);
    P = end_effector_pos;
    s = 12;
    claw_local = [0 0 0; 0 s 0; s s 0; 0 s 0; 0 0 0;
         0 0 0; 0 -s 0; s -s 0; 0 -s 0; 0 0 0;
         0 0 0; 0 0 s; s 0 s; 0 0 s; 0 0 0;
         0 0 0; 0 0 -s; s 0 -s; 0 0 -s; 0 0 0];

    claw_world = (R * claw_local')' + P';
    plot3(ax, claw_world(:,1), claw_world(:,2), ...
        claw_world(:,3), 'b', 'LineWidth', 5);
    
    % Wrist joint circle
    plot3(ax, P(1), P(2), P(3), 'wo', 'MarkerSize', ...
        10, 'MarkerFaceColor', 'w', 'LineWidth', 2);

    %% End-Effector Direction Arrow
    R = T_final(1:3,1:3);
    arrow_len = 25;
    dir = R(:,1);
    quiver3(ax, end_effector_pos(1), end_effector_pos(2), ...
        end_effector_pos(3), arrow_len*dir(1), ...
        arrow_len*dir(2), arrow_len*dir(3), 'r', ...
        'LineWidth', 3, 'MaxHeadSize', 0.8);
    title(ax, ['6DOF Robot ' char(fig_title)]);
    
    if ~silent
        fprintf('\n');
        fprintf('# 6DOF %s\n', fig_title);
        fprintf('Calculated Joint Angles (in deg)\n');
        for i = 1:6
            fprintf('Q%d = %8.2f deg\n', i, q_sol(i));
        end
        % Joint Positions
        fprintf('\nJoint Positions (mm)\n');
        for i = 1:size(all_positions,2)
            fprintf('P%d : X = %8.2f   Y = %8.2f   Z = %8.2f\n', ...
                i-1, all_positions(1,i), ...
                all_positions(2,i), ...
                all_positions(3,i));
        end
        
        % End-Effector Position
        fprintf('\nEnd-Effector Position\n');
        fprintf('X = %8.2f mm\n', end_effector_pos(1));
        fprintf('Y = %8.2f mm\n', end_effector_pos(2));
        fprintf('Z = %8.2f mm\n', end_effector_pos(3));
        fprintf('\nTransformation Matrix T_final\n');
        disp(T_final);
    end
end