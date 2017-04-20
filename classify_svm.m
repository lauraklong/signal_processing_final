function [ Azs, trainpredictions, accuracies ] = classify_svm( subjects,dataversion,channels,timewindow,ploteachsubject)
classifiername = 'svm';

%% Load data and build classifier

% Set up variables
Azs = zeros(1,length(subjects));
accuracies = Azs;
trainpredictions = cell(1,length(subjects));

% Loop over subjects
for i = 1:length(subjects);
    
    clear EEG; clear X_EEG_TRAIN; clear X_EEG_TEST; clear Y_EEG_TRAIN % clear variables
    
    % Load data
    LOAD_PATH = fullfile('data', ['data_v' num2str(dataversion)], ['Subject_', num2str(subjects(i)), '.mat']);
    load(LOAD_PATH);
    
    % Shuffle training data and labels (so initialization is not biased)
    neworder = randperm(length(Y_EEG_TRAIN));
    truelabels = Y_EEG_TRAIN(neworder);
    reordertrain = X_EEG_TRAIN(:,:,neworder);
    
    % Take average over particular time window for selected channels
    if isempty(channels)
        x = squeeze(mean(reordertrain(:,timewindow,:),2));
    else
        x = squeeze(mean(reordertrain(channels,timewindow,:),2));
    end
    x = x';
    
    traindata = x; 
    trainlabels = truelabels;
    
     % Run SVM
    leftout_predictions = zeros(1,size(traindata,1));
    for j = 1:size(traindata,1)
        disp(['Leaving out sample ' num2str(j)]);
        
        % Define left-out data
        leftoutdata = traindata(j,:);
        
        % Delete the left-out sample from data and labels
        remainingdata = traindata;
        remainingdata(j,:) = [];
        remaininglabels = trainlabels;
        remaininglabels(j) = [];
        
        % Find discriminator and coefficients
        model_svm = fitcsvm(remainingdata,remaininglabels);
        
        % Predict this sample's result
        leftout_predictions(j) = predict(model_svm,leftoutdata);
    end

    
    idx = leftout_predictions';
   
    % Find ROC curve; for k-means, sometimes the clustering will pick the wrong label for the classes; if this is the case (Az<.5), swap them and replot
    [Az,accuracy] = plotROCCurve(truelabels,idx,ploteachsubject,classifiername);
    if Az < .5
        idx = logical(idx);
        idx = ~idx;
        idx = double(idx);
        [Az,accuracy] = plotROCCurve(truelabels,idx,ploteachsubject,classifiername);
    end
    
    % Store each subject's data
    Azs(i) = Az;
    trainpredictions{i} = idx; 
    accuracies(i) = accuracy;
    
end


end

