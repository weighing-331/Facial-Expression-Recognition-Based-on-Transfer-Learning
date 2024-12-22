clc; clear; close all;

% 加載完整數據集
[merchImagesTrain, merchImagesTest] = merchData();

% 隨機抽樣
rng(123); % 固定隨機數種子
[smallTrainSet, ~] = splitEachLabel(merchImagesTrain, 100 / numel(merchImagesTrain.Files), 'randomized');
[smallTestSet, ~] = splitEachLabel(merchImagesTest, 20 / numel(merchImagesTest.Files), 'randomized');

% 調整影像大小 (灰階圖像)
inputSize = [28, 28, 1]; % LeNet 的輸入大小
augimdsTrain = augmentedImageDatastore(inputSize(1:2), smallTrainSet);
augimdsTest = augmentedImageDatastore(inputSize(1:2), smallTestSet);

% 構建 LeNet 模型
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(5, 20, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    convolution2dLayer(5, 50, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    fullyConnectedLayer(500, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(numel(categories(smallTrainSet.Labels)), 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% 訓練選項
epochs = [5, 10, 15, 20];
results = [];

for i = 1:length(epochs)
    for trial = 1:5
        disp("Epoch: " + epochs(i) + ", Trial: " + trial);

        % 設定訓練選項
        options = trainingOptions('sgdm', ...
            'MiniBatchSize', 64, ...
            'MaxEpochs', epochs(i), ...
            'InitialLearnRate', 0.0001, ...
            'Plots', 'none', ...
            'ExecutionEnvironment', 'auto');

        % 訓練模型
        netTransfer = trainNetwork(augimdsTrain, layers, options);

        % 測試模型性能
        predictedLabels = classify(netTransfer, augimdsTest);
        testLabels = smallTestSet.Labels;
        accuracy = mean(predictedLabels == testLabels);

        % 計算混淆矩陣
        confMat = confusionmat(testLabels, predictedLabels);
        TP = confMat(1, 1);
        FP = confMat(2, 1);
        FN = confMat(1, 2);
        TN = confMat(2, 2);

        % Precision, Recall, F1-Score
        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        f1_score = 2 * (precision * recall) / (precision + recall);

        % 計算 ROC-AUC
        try
            predictedScores = predict(netTransfer, augimdsTest);
            [~, ~, ~, auc] = perfcurve(testLabels, predictedScores(:, 1), 'happy');
        catch
            auc = NaN;
        end

        % 記錄結果
        results = [results; epochs(i), trial, accuracy, precision, recall, f1_score, auc];
    end
end

% 將結果寫入 Excel
resultsTable = array2table(results, 'VariableNames', {'Epoch', 'Trial', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'});
writetable(resultsTable, 'lenet_results.xlsx');
disp("Results saved to 'lenet_results.xlsx'.");