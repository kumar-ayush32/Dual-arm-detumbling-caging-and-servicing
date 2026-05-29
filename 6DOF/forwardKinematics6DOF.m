robot = model_6DOF();
q_deg = [0, 45, 45, 45, 45, 0];
silent = false;

[T_final, all_positions] = compute_fk(robot, q_deg);
robot_plot(T_final, all_positions, silent);
fprintf('Given Joint Angles (in deg)\n');
for i = 1:6
    fprintf('Q%d = %8.2f deg\n', i, q_deg(i));
end

function robot_plot(T_final, all_positions, silent)
    end_effector_pos = T_final(1:3,4);
    figFK = figure('Name', '6DOF Robot', 'NumberTitle', 'off', 'Color', 'black', ...
        'Position', [100 100 700 650]);
    ax = axes('Parent', figFK);
    hold(ax, 'on');
    axis(ax, 'equal');
    grid(ax, 'on');
    view(ax, 3);
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'Z');

    x = all_positions(1,:);
    y = all_positions(2,:);
    z = all_positions(3,:);

    % Plot Links
    plot3(ax, x, y, z, '-o', ...
        'Color', [0 0.4470 0.7410], ...
        'LineWidth', 2.5, ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', 'r');

    % Solid Base Marker
    plot3(ax, 0,0,0, 's', 'MarkerSize', 18, ...
        'MarkerFaceColor', 'y', ...
        'MarkerEdgeColor', 'y', ...
        'LineWidth', 2);

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

    %% Axis Limits
    max_range = max([max(abs(x)), max(abs(y)), max(abs(z)), 200]);
    axis(ax, [-max_range max_range -max_range max_range -max_range max_range]);
    title(ax, '6DOF Robot Arm');
    hold(ax, 'off');

    if ~silent
        fprintf('\n');
        fprintf('# 6DOF FORWARD KINEMATICS\n');
        
        
        %% Joint Positions
        fprintf('\nJoint Positions (mm)\n');
        for i = 1:size(all_positions,2)
            fprintf('P%d : X = %8.2f   Y = %8.2f   Z = %8.2f\n', ...
                i-1, all_positions(1,i), ...
                all_positions(2,i), ...
                all_positions(3,i));
        end
        
        %% End-Effector Position
        fprintf('\nEnd-Effector Position\n');
        fprintf('X = %8.2f mm\n', end_effector_pos(1));
        fprintf('Y = %8.2f mm\n', end_effector_pos(2));
        fprintf('Z = %8.2f mm\n', end_effector_pos(3));
        fprintf('\nTransformation Matrix T_final\n');
        disp(T_final);
    end
end