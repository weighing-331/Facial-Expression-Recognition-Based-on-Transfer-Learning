function [merchImagesTrain, merchImagesTest] = merchData()
    % 設定測試數據的路徑
    testFolder = fullfile(pwd, 'test');  % 當前目錄下的 test 資料夾

    % 加載測試資料集
    if isfolder(testFolder)
        merchImagesFull = imageDatastore(testFolder, ...
            'IncludeSubfolders', true, ...
            'LabelSource', 'foldernames'); % 標籤來自資料夾名稱
    else
        error("Testing folder does not exist: " + testFolder);
    end

    % 將測試資料分割為訓練和測試集
    trainRatio = 0.7; % 可根據需要調整比例
    [merchImagesTrain, merchImagesTest] = splitEachLabel(merchImagesFull, trainRatio, 'randomized');

    % 檢查生成的標籤
    disp("Training labels:");
    disp(unique(merchImagesTrain.Labels));
    disp("Testing labels:");
    disp(unique(merchImagesTest.Labels));
    disp("Training samples: " + numel(merchImagesTrain.Files));
    disp("Testing samples: " + numel(merchImagesTest.Files));
end
