clc; clear; close all;

% 加載完整數據集
[merchImagesTrain, merchImagesTest] = merchData();

% 隨機抽樣
rng(123); % 固定隨機數種子
[smallTrainSet, ~] = splitEachLabel(merchImagesTrain, 1000 / numel(merchImagesTrain.Files), 'randomized');
[smallTestSet, ~] = splitEachLabel(merchImagesTest, 200 / numel(merchImagesTest.Files), 'randomized');

% 調整影像大小
inputSize = [227, 227, 3]; % SqueezeNet 的輸入大小
augimdsTrain = augmentedImageDatastore(inputSize(1:2), smallTrainSet, 'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2), smallTestSet, 'ColorPreprocessing', 'gray2rgb');

% 訓練選項
epochs = [5, 10, 15, 20];
results = [];

for i = 1:length(epochs)
    for trial = 1:5
        disp("Epoch: " + epochs(i) + ", Trial: " + trial);
        
        % 設定 SqueezeNet
        net = squeezenet;
        lgraph = layerGraph(net);
        numClasses = numel(categories(smallTrainSet.Labels));
        
        % 移除原始輸出層及相關未連接的層
        lgraph = removeLayers(lgraph, {'conv10', 'relu_conv10', 'pool10', 'prob', 'ClassificationLayer_predictions'});

        % 添加新層
        newConvLayer = convolution2dLayer(1, numClasses, 'Name', 'new_conv', ...
            'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
        newSoftmaxLayer = softmaxLayer('Name', 'new_softmax');
        newClassificationLayer = classificationLayer('Name', 'new_classification');
        avgPoolLayer = globalAveragePooling2dLayer('Name', 'global_avg_pool');

        % 正確連接新層
        lgraph = addLayers(lgraph, newConvLayer);
        lgraph = addLayers(lgraph, avgPoolLayer);
        lgraph = addLayers(lgraph, newSoftmaxLayer);
        lgraph = addLayers(lgraph, newClassificationLayer);
        lgraph = connectLayers(lgraph, 'drop9', 'new_conv');
        lgraph = connectLayers(lgraph, 'new_conv', 'global_avg_pool');
        lgraph = connectLayers(lgraph, 'global_avg_pool', 'new_softmax');
        lgraph = connectLayers(lgraph, 'new_softmax', 'new_classification');

        % 設定訓練選項
        options = trainingOptions('sgdm', ...
            'MiniBatchSize', 4, ...
            'MaxEpochs', epochs(i), ...
            'InitialLearnRate', 0.0001, ...
            'Plots', 'none', ...
            'ExecutionEnvironment', 'auto');

        % 訓練模型
        netTransfer = trainNetwork(augimdsTrain, lgraph, options);

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
writetable(resultsTable, 'model_results.xlsx');
disp("Results saved to 'model_results.xlsx'.");
