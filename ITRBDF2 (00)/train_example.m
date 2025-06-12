clear all
close all
clc
rng(2025)

%% 1) Generate FHN Data
% Parameters of FHN system
a = 0.7;
b = 0.8;
epsilon = 0.08;
I = 0.5;

% Define FHN ODE
fhn = @(t, x) [
    x(1) - (1/3)*x(1)^3 - x(2) + I;
    epsilon * (x(1) + a - b*x(2))
];

x0 = [0; 0];
T = 50;
numTimeSteps = 1000;
t = linspace(0, T, numTimeSteps);
odeOptions = odeset('RelTol', 1e-9, 'AbsTol', 1e-10);
[~, x] = ode45(fhn, t, x0, odeOptions);
x = x';
%% 2) Define Neural Network
inputSize = size(x,1);  % 2 for FHN
hiddenSize = 32;
outputSize = inputSize;

neuralOdeLayers = [
    fullyConnectedLayer(hiddenSize)
    tanhLayer
    fullyConnectedLayer(hiddenSize)
    tanhLayer
    fullyConnectedLayer(outputSize)
];

neuralOdeInternalDlnetwork = dlnetwork(neuralOdeLayers,'Initialize',false);

% Custom neural ODE layer
neuralOdeInternalTimesteps = 40;
dt = t(2) - t(1);
neuralOdeLayerName = 'neuralOde';
customNeuralOdeLayer = neuralOdeLayer(neuralOdeInternalDlnetwork, neuralOdeInternalTimesteps, dt, neuralOdeLayerName);

dlnet = dlnetwork(customNeuralOdeLayer, 'Initialize', false);
dlnet = initialize(dlnet, dlarray(ones(inputSize,1),'CB'));

%% 3) Training Setup
gradDecay = 0.9;
sqGradDecay = 0.999;
learnRate = 0.001;
numIter = 2500;
miniBatchSize = 200;
plotFrequency = 50;
plots = "training-progress";
lossHistory = [];

if plots == "training-progress"
    figure(1)
    clf
    title('Training Loss');
    lossline = animatedline;
    xlabel('Iteration')
    ylabel("Loss")
    grid on
end

%% 4) Training Loop
averageGrad = [];
averageSqGrad = [];
lossHistory = zeros(numIter, 1); % preallocate for speed
trainingTimesteps = 1:numTimeSteps;
start = tic;

for iter=1:numIter
    [dlx0, targets] = createMiniBatch(numTimeSteps, neuralOdeInternalTimesteps, miniBatchSize, x);
    [grads,loss] = dlfeval(@modelGradients,dlnet,dlx0,targets);
    [dlnet,averageGrad,averageSqGrad] = adamupdate(dlnet,grads,averageGrad,averageSqGrad,iter,...
        learnRate,gradDecay,sqGradDecay);
    
    currentLoss = extractdata(loss);
    lossHistory(iter) = currentLoss;
    if plots == "training-progress"
        addpoints(lossline, iter, currentLoss);
        drawnow
    end

    if mod(iter, plotFrequency) == 0
        figure(2)
        clf
        internalNeuralOdeLayer = dlnet.Layers(1);
        dlnetODEFcn = @(t,y) evaluateODE(internalNeuralOdeLayer, y);
        [~,y] = ode45(dlnetODEFcn, [t(1) t(end)], x0, odeOptions);
        y = y';
        plot(x(1,:), x(2,:), 'r--', y(1,:), y(2,:), 'b-')
        grid on
        xlabel('v'); ylabel('w');
        title("Iter = " + iter + ", loss = " + num2str(currentLoss))
        legend('Ground truth','Predicted')
    end
end
toc(start);

save('FHN_NeuralODE_TrainingLoss_ITRBDF2.mat', 'lossHistory');
save('trainedFHN_NeuralODE_ITRBDF2.mat', 'dlnet');
%% 5) Evaluation
figure
initialConditions = {
    [-1; 1], '[-1 1]';
    [1; -1], '[1 -1]';
    [0; 0], '[0 0]';
    [2; 2], '[2 2]'
};

for i = 1:4
    x0Test = initialConditions{i,1};
    label = initialConditions{i,2};
    [xPred, xTrue, err] = predictWithode45(dlnet, fhn, t, x0Test, odeOptions);
    subplot(2,2,i)
    plotTrueAndPredictedSolutions(xTrue, xPred, err, label);
end

% Helper Functions
function [gradients,loss] = modelGradients(dlnet, dlX0, targets)
    dlX = forward(dlnet,dlX0);
    loss = sum(abs(dlX - targets), 'all') / numel(dlX);
    gradients = dlgradient(loss,dlnet.Learnables);
end

function [dlX0, dlT] = createMiniBatch(numTimesteps, numTimesPerObs, miniBatchSize, X)
    s = randperm(numTimesteps - numTimesPerObs, miniBatchSize);
    dlX0 = dlarray(X(:, s),'CB');
    dlT = zeros([size(dlX0,1) miniBatchSize numTimesPerObs]);
    for i = 1:miniBatchSize
        dlT(:, i, 1:numTimesPerObs) = X(:, s(i):(s(i) + numTimesPerObs - 1));
    end
end

function [xPred, xTrue, error] = predictWithode45(dlnet,fhn,tPred,x0Pred,odeOptions)
    [~, xTrue] = ode45(fhn, tPred, x0Pred, odeOptions);
    internalNeuralOdeLayer = dlnet.Layers(1);
    dlnetODEFcn = @(t,y) evaluateODE(internalNeuralOdeLayer, y);
    [~,xPred] = ode45(dlnetODEFcn, tPred, x0Pred, odeOptions);
    error = mean(abs(xTrue - xPred), 'all');
end

function plotTrueAndPredictedSolutions(xTrue, xPred, err, x0Str)
    plot(xTrue(:,1),xTrue(:,2),'r--',xPred(:,1),xPred(:,2),'b-','LineWidth',1)
    title("x_0 = " + x0Str + ", err = " + num2str(err))
    xlabel('v')
    ylabel('w')
    legend('Ground truth', 'Predicted')
    grid on
end