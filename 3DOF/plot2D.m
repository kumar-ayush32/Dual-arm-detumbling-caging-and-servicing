function plot2D(q_solution, link_length, ax, color)
    L1 = link_length(1);
    L2 = link_length(2);
    L3 = link_length(3);
    q1 = q_solution(1);
    q2 = q_solution(2);
    q3 = q_solution(3);
    a1 =  q1;
    a2 =  q1 + q2;
    a3 =  q1 + q2 + q3;

    P0 = [0; 0];
    P1 = P0 + L1 * [cos(a1); sin(a1)];
    P2 = P1 + L2 * [cos(a2); sin(a2)];
    P3 = P2 + L3 * [cos(a3); sin(a3)];
    fprintf('\n');
    fprintf('# Inverse Kinematics Results  (3-DOF Planar 2D)\n');
    fprintf('# Joint Angles (in deg):   q1=%7.2f   q2=%7.2f   q3=%7.2f\n', ...
        rad2deg(q1), rad2deg(q2), rad2deg(q3));
    fprintf('\n');

    fprintf('  Joint Positions:\n');
    fprintf('\tBase   : (%.4f,  %.4f) m\n', P0(1), P0(2));
    fprintf('\tElbow  : (%.4f,  %.4f) m\n', P1(1), P1(2));
    fprintf('\tWrist  : (%.4f,  %.4f) m\n', P2(1), P2(2));
    fprintf('\tEnd-Eff: (%.4f,  %.4f) m\n', P3(1), P3(2));

    fprintf('\n');
    fprintf('  End-Effector:\n');
    fprintf('\tX     = %8.4f m\n', P3(1));
    fprintf('\tY     = %8.4f m\n', P3(2));
    fprintf('\tAngle = %8.2f deg  (orientation in XY plane)\n', rad2deg(a3));

    r_max = L1 + L2 + L3;
    r_min = abs(L1 - L2 - L3);
    
    %% 2D Visualization
    % Workspace circles
    theta_c = linspace(0, 2*pi, 300);
    plot(ax, r_max * cos(theta_c), r_max * sin(theta_c), ...
        '--', 'Color', [0.75 0.75 0.75], 'LineWidth', 1.0, ...
        'DisplayName', sprintf('Max reach (%.3f m)', r_max));
    if r_min > 0
        plot(ax, r_min * cos(theta_c), r_min * sin(theta_c), ...
            ':', 'Color', [0.85 0.75 0.75], 'LineWidth', 1.0, ...
            'DisplayName', sprintf('Min reach (%.3f m)', r_min));
    end

    % Links
    pts = [P0, P1, P2, P3];
    plot(ax, pts(1,:), pts(2,:), ...
        '-o', ...
        'Color',           color, ...
        'LineWidth',       3.0, ...
        'MarkerSize',      8, ...
        'MarkerFaceColor', [0.15 0.35 0.70], ...
        'DisplayName',     'Links');

    % Joint labels
    labels = {'Base','J1','J2','J3 (EE)'};
    offsets = [0.02, 0.02; 0.02, 0.02; 0.02, 0.02; 0.02, 0.02];
    for k = 1:4
        text(ax, pts(1,k)+offsets(k,1), pts(2,k)+offsets(k,2), ...
            labels{k}, ...
            'FontSize',   9, ...
            'FontWeight', 'bold', ...
            'Color',      color);
    end

    % ---------- Fixed Claw End-Effector ----------
    
    % Claw dimensions
    claw_w = 0.03;
    claw_h = 0.06;
    claw_pts = [0 0; 0 claw_h/2; claw_w claw_h/2; 0 claw_h/2;
        0 -claw_h/2; claw_w -claw_h/2];
    R = [cos(a3) -sin(a3);
         sin(a3)  cos(a3)];
    claw_world = (R * claw_pts')' + P3';
    
    plot(ax, claw_world(:,1), claw_world(:,2), ...
        'k', 'LineWidth', 3, 'Color', 'r');
    scatter(ax, P3(1), P3(2), 80, ...
        'w', 'filled', 'MarkerEdgeColor', 'k', ...
        'LineWidth', 2);

    % EE orientation arrow
    arrow_len = 0.05;
    quiver(ax, P3(1), P3(2), ...
           arrow_len*cos(a3), arrow_len*sin(a3), ...
           0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.8, ...
           'DisplayName', 'EE orientation');

    % Base marker
    scatter(ax, 0, 0, 120, 'ks', 'filled', 'DisplayName', 'Base');
    title(ax, 'IK 2D', 'FontSize', 13);
    xlabel(ax, 'X (m)');
    ylabel(ax, 'Y (m)');
    % legend(ax, 'Location', 'northeast');
end