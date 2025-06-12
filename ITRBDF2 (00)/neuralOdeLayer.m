classdef neuralOdeLayer < nnet.layer.Layer & nnet.layer.Formattable
    % neuralOdeLayer Custom Neural ODE layer
    % 
    % This custom layer learns the trainable parameters of an internal
    % dlnetwork object by backpropagating through the operations of an
    % explicit 4th-order Runge-Kutta numerical solver.
    
    %   Copyright 2021 The MathWorks, Inc.

    properties
        % Computational grid for RK-4 method
        Timesteps
    end
    
    properties (Learnable)
        % Learnable property
        InternalDlnet
    end
    
    methods
        % Constructor
        function this = neuralOdeLayer(dlnet,Nt,h,name)
            % Declare layer timesteps
            this.Timesteps = (1:Nt)*h;
            
            this.Name = name;
            this.InternalDlnet= dlnet;
        end

        function dlnetModelFcn = getInternalDlnetAsFcn(this)
            % Create a function dlnetModelFcn which can be given to an ODE
            % solver of the form x' = fcn( t, x ).
            dlnetModelFcn = @(t,x)this.InternalDlnet.forward(x);
        end

        function x = evaluateODE(this,x)
            x = dlarray(x,'CB');
            x = this.InternalDlnet.predict(x);
            x = extractdata(x);
            x = cast(x,'double');
        end

        function h = predict(this, X)
            dlnetModelFcn = this.getInternalDlnetAsFcn();
            h = itrbdf2Batch(dlnetModelFcn, this.Timesteps, X);
            h = dlarray(h, 'CBT');
        end
    end
end

function x = itrbdf2Batch(fcn, t, x0)
% TR-BDF2 ODE solver for batched inputs
%
% Inputs:
%   fcn        - function handle f(t, x)
%   t          - time vector (1 x T)
%   x0         - initial condition (C x B)
%   useClassic - if true, uses classic TR-BDF2 (default: false)
%
% Output:
%   x - solution (C x B x T)

% Constants for TR-BDF2
gamma = 2 - sqrt(2);  % γ ≈ 0.5858

% Dimensions
[C, B] = size(x0);
T = numel(t);
x = cat(3, x0, zeros(C, B, T - 1));  % Allocate result array

% Time stepping
for ii = 1:(T - 1)
    dt = t(ii+1) - t(ii);
    t_now = t(ii);
    x_now = x(:, :, ii);

    % Evaluate xp = f(t, x)
    xp = fcn(t_now, x_now);

    % Stage 1 (TR-like)
    t_gamma = t_now + gamma * dt;
    xk = x_now + gamma * dt * xp;
    fk_gamma = fcn(t_gamma, xk);
    fk_gamma = 0.5 * (fk_gamma + xp);  % Comment out for Classic TR-BDF2

    xk_gamma = x_now + gamma * (dt / 2) * (xp + fk_gamma);

    % Stage 2 (BDF2-like)
    t_next = t_now + dt;
    x1 = x_now + dt * xp;
    fk1 = fcn(t_next, x1);
    fk1 = 0.5 * (fk1 + xp);  % Comment out for Classic TR-BDF2

    % Final update
    x(:, :, ii + 1) = (1 / (gamma * (2 - gamma))) * xk_gamma ...
                    - ((1 - gamma)^2 / (gamma * (2 - gamma))) * x_now ...
                    + ((1 - gamma) / (2 - gamma)) * dt * fk1;
end

end