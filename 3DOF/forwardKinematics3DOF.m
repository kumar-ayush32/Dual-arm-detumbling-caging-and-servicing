function T_final = forwardKinematics3DOF(q, silent, link_length)

    % Default joint angles
    if nargin < 1 || isempty(q)
        q = [45,60,30];
    end

    % Default silent mode
    if nargin < 2 || isempty(silent)
        silent = false;
    end

    % Default link lengths
    if nargin < 3 || isempty(link_length)
        link_length = [0.3 0.2 0.2];
    end

    % Extract link lengths
    L1 = link_length(1);
    L2 = link_length(2);
    L3 = link_length(3);
    
    % q = [q1; q2; q3], Joint angles in radians (shoulder, elbow, wrist)
    q = deg2rad(q(:));
    q1 = q(1);  q2 = q(2);  q3 = q(3);
    a1 =  q1;
    a2 =  q1 + q2;
    a3 =  q1 + q2 + q3;

    P0 = [0; 0];
    P1 = P0 + L1 * [cos(a1); sin(a1)];
    P2 = P1 + L2 * [cos(a2); sin(a2)];
    P3 = P2 + L3 * [cos(a3); sin(a3)];

    T_final = [ cos(a3), -sin(a3), 0,  P3(1);
                sin(a3),  cos(a3), 0,  P3(2);
                0,        0,       1,  0;
                0,        0,       0,  1    ];

    if ~silent
        fprintf('\n');
        fprintf('# Forward Kinematics Results  (3-DOF Planar 2D)\n');
        fprintf('  Joint Angles (deg):   q1=%7.2f   q2=%7.2f   q3=%7.2f\n', ...
            rad2deg(q1), rad2deg(q2), rad2deg(q3));
        fprintf('\n');

        fprintf('  Transformation Matrix T_0→EE:\n');
        for r = 1:4
            fprintf('    [ ');
            fprintf('%10.5f  ', T_final(r,:));
            fprintf(']\n');
        end
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
        figFK = figure('Name', 'Forward Kinematics – 3DOF Planar 2D', ...
            'NumberTitle', 'off', 'Color', 'black', 'Position', [100 100 700 650]);

        ax = axes('Parent', figFK);
        hold(ax, 'on');
        axis(ax, 'equal');
        grid(ax, 'on');

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
            'Color',           [0.15 0.35 0.70], ...
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
                'Color',      [0.15 0.35 0.70]);
        end

        % End-effector marker
        scatter(ax, P3(1), P3(2), 180, 'r', 'filled', ...
            'DisplayName', 'End-Effector');

        % EE orientation arrow
        arrow_len = 0.05;
        quiver(ax, P3(1), P3(2), ...
               arrow_len*cos(a3), arrow_len*sin(a3), ...
               0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.8, ...
               'DisplayName', 'EE orientation');

        % Base marker
        scatter(ax, 0, 0, 120, 'ks', 'filled', 'DisplayName', 'Base');

        title(ax, sprintf('FK 2D  |  q = [%.1f°, %.1f°, %.1f°]', ...
              rad2deg(q1), rad2deg(q2), rad2deg(q3)), ...
              'FontSize', 13);
        xlabel(ax, 'X (m)');
        ylabel(ax, 'Y (m)');
        legend(ax, 'Location', 'northeast');
        hold(ax, 'off');
    end
end
