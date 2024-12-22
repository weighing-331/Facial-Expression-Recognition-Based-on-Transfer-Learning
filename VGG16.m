clc; clear; close all;

% 加載完整數據集
[merchImagesTrain, merchImagesTest] = merchData();

% 隨機抽樣
rng(123); % 固定隨機數種子
[smallTrainSet, ~] = splitEachLabel(merchImagesTrain, 100 / numel(merchImagesTrain.Files), 'randomized');
[smallTestSet, ~] = splitEachLabel(merchImagesTest, 20 / numel(merchImagesTest.Files), 'randomized');

% 調整影像大小
inputSize = [224, 224, 3];
augimdsTrain = augmentedImageDatastore(inputSize(1:2), smallTrainSet, 'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2), smallTestSet, 'ColorPreprocessing', 'gray2rgb');

% 構建 VGG16 模型
net = vgg16;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(smallTrainSet.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer
    classificationLayer];

% 設定訓練選項 auto or cpu/ epoch 5, 10, 15, 20
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 4, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 0.0001, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'cpu');

% 微調模型
netTransfer = trainNetwork(augimdsTrain, layers, options);

% 測試模型性能
predictedLabels = classify(netTransfer, augimdsTest);
testLabels = smallTestSet.Labels;
accuracy = mean(predictedLabels == testLabels);
disp("Accuracy on small dataset: " + accuracy);



% 獲取預測標籤
predictedLabels = classify(netTransfer, augimdsTest);

% 計算混淆矩陣
confMat = confusionmat(testLabels, predictedLabels);
disp("Confusion Matrix:");
disp(confMat);

% 提取混淆矩陣中的值
TP = confMat(1, 1); % True Positive
FP = confMat(2, 1); % False Positive
FN = confMat(1, 2); % False Negative
TN = confMat(2, 2); % True Negative

% Precision, Recall, F1-Score
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

disp("Precision: " + precision);
disp("Recall: " + recall);
disp("F1-Score: " + f1_score);

% 計算 ROC-AUC
try
    predictedScores = predict(netTransfer, augimdsTest); % 獲取預測概率
    [~, ~, ~, auc] = perfcurve(testLabels, predictedScores(:, 1), 'happy');
    disp("ROC-AUC: " + auc);

    % 繪製 ROC 曲線
    [X, Y, ~, ~] = perfcurve(testLabels, predictedScores(:, 1), 'happy');
    figure;
    plot(X, Y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title('ROC Curve (Happy vs Sad)');
catch ME
    warning("Error calculating or plotting ROC-AUC: " + ME.message);
end

% 繪製混淆矩陣
cm = confusionchart(testLabels, predictedLabels);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'VGG16 Confusion Matrix (Happy vs Sad)';