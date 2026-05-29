function q_sol = inverseKinematics6DOF(robot, target_pos, q0)
    MAX_ITER       = 5000;
    POS_TOL        = 1e-3;

    DELTA          = 1e-4;

    LAMBDA_MAX     = 0.5;
    LAMBDA_MIN     = 1e-4;
    ERR_SCALE      = 10.0;

    STEP_LIMIT_DEG = 2.0;
    STEP_LIMIT_RAD = deg2rad(STEP_LIMIT_DEG);

    FREE_JOINTS  = [1, 2, 3, 4, 5, 6];
    FIXED_JOINTS = [];

    n_free = numel(FREE_JOINTS);
    q = deg2rad(q0(:)');
    q_fixed_vals = q(FIXED_JOINTS);

    % Joint limits for free joints only
    q_min = deg2rad(-180 * ones(1, n_free));
    q_max = deg2rad( 180 * ones(1, n_free));

    target_pos = target_pos(:);
    converged  = false;

    for iter = 1:MAX_ITER
        q(FIXED_JOINTS) = q_fixed_vals;

        [T, ~]    = compute_fk(robot, rad2deg(q));
        p_current = T(1:3, 4);

        e        = target_pos - p_current;
        err_norm = norm(e);

        if err_norm < POS_TOL
            converged = true;
            break;
        end

        J = zeros(3, n_free);

        for k = 1:n_free
            i = FREE_JOINTS(k);

            q_plus        = q;
            q_minus       = q;
            q_plus(i)     = q_plus(i)  + DELTA;
            q_minus(i)    = q_minus(i) - DELTA;

            % Re-enforce fixed joints after perturbation
            q_plus(FIXED_JOINTS)  = q_fixed_vals;
            q_minus(FIXED_JOINTS) = q_fixed_vals;

            [Tp, ~] = compute_fk(robot, rad2deg(q_plus));
            [Tm, ~] = compute_fk(robot, rad2deg(q_minus));

            J(:, k) = (Tp(1:3,4) - Tm(1:3,4)) / (2 * DELTA);
        end

        % ADAPTIVE DAMPING
        lambda = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * ...
                 min(err_norm / ERR_SCALE, 1.0);

        % DAMPED LEAST SQUARES
        dq_free = J' * ((J * J' + lambda^2 * eye(3)) \ e);
        dq_free = dq_free';

        dq_free = max(-STEP_LIMIT_RAD, min(STEP_LIMIT_RAD, dq_free));
        q(FREE_JOINTS) = q(FREE_JOINTS) + dq_free;
        q(FREE_JOINTS) = max(q_min, min(q_max, q(FREE_JOINTS)));
    end

    q(FIXED_JOINTS) = q_fixed_vals;
    q_sol = wrapTo180(rad2deg(q));
    [T_final, ~] = compute_fk(robot, q_sol);
    final_error  = norm(target_pos - T_final(1:3,4));

    if ~converged
        warning('inverseKinematics6DOF: did NOT converge after %d iterations. Final error = %.4f mm.', ...
                MAX_ITER, final_error);
    else
        fprintf('[IK] Converged in %d iterations. Final error = %.4e mm\n', iter, final_error);
    end
end