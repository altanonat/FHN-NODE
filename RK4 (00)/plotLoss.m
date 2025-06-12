clear all
close all
clc
rk4 = load('FHN_NeuralODE_TrainingLoss_RK4.mat', 'lossHistory');
figure(2)
plot(rk4.lossHistory,'-k')
hold on
xlabel('Iteration')
ylabel('Loss')
title('Training Loss Over Time')
grid on

%% 5) Evaluation
rk4dl     = load('trainedFHN_NeuralODE_RK4.mat', 'dlnet');
% Parameters
a = 0.7;
b = 0.8;
epsilon = 0.08;
I = 0.5;

fhn = @(t, x) [
    x(1) - (1/3)*x(1)^3 - x(2) + I;
    epsilon * (x(1) + a - b*x(2))
];

x0 = [0; 0];
label = '[0 0]';
T = 50;
numTimeSteps = 1000;
t = linspace(0, T, numTimeSteps);
odeOptions = odeset('RelTol', 1e-9, 'AbsTol', 1e-10);
fhn = @(t, x) [
    x(1) - (1/3)*x(1)^3 - x(2) + I;
    epsilon * (x(1) + a - b*x(2))
];

% Predict and plot
[xPred, xTrue, err] = predictWithOde45(rk4dl.dlnet, fhn, t, x0, odeOptions);
figure
plotTrueAndPredictedSolutions(xTrue, xPred, err, label);

%% Helper Functions
function [xPred, xTrue, error] = predictWithOde45(dlnet,fhn,tPred,x0Pred,odeOptions)
    [~, xTrue] = ode45(fhn, tPred, x0Pred, odeOptions);
    internalNeuralOdeLayer = dlnet.Layers(1);
    dlnetODEFcn = @(t,y) evaluateODE(internalNeuralOdeLayer, y);
    [~,xPred] = ode45(dlnetODEFcn, tPred, x0Pred, odeOptions);
    error = mean(abs(xTrue - xPred), 'all');
end

function plotTrueAndPredictedSolutions(xTrue, xPred, err, x0Str)
    % Convert width to inches (5.3 cm ≈ 2.087 inches)
    width_in = 5.3 / 2.54;
    height_in = width_in * 0.75;  % Adjust height (e.g., 4:3 ratio)

    fig = figure;
    set(fig, 'Units', 'inches', 'Position', [1 1 width_in height_in])
    set(fig, 'PaperUnits', 'inches', 'PaperPosition', [0 0 width_in height_in])

    % Plotting
    plot(xTrue(:,1), xTrue(:,2), 'k--', ...
         xPred(:,1), xPred(:,2), 'Color', [0.5 0.5 0.5], 'LineWidth', 1)

    % Formatting
    ax = gca;
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 8)
    xlabel('v', 'FontName', 'Times New Roman', 'FontSize', 8)
    ylabel('w', 'FontName', 'Times New Roman', 'FontSize', 8)
    title("x_0 = " + x0Str + ", err = " + num2str(err), ...
        'FontName', 'Times New Roman', 'FontSize', 8)

    % legend('Ground truth', 'Predicted - RK4', ...
    %     'Location', 'best', 'FontName', 'Times New Roman', 'FontSize', 8)

    grid on
end
