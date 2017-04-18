%% Initial setup
clear all; clc;
classifiername = 'kmeans'; % classifier name
classifiertag = ''; % tag for saving
disp(['running ' classifiername]);

%% SETTINGS

% saving/plotting 
savedata = 1; % will save data to results folder
ploteachsubject = 0; % will plot ROC for every subject

% which data to use
PROCESSED_DATA_VERSION = 0; % which data to load (currently only works with version 0; needs the X_EEG_TRAIN variable to be intact as chanxtimexepoch)
subjects = 1:8; % which subjects to run

% for classifier
timewindow = 250:500;
channels = [];

%% Load data and build classifier

% Set up variables
Azs = zeros(1,length(subjects));
accuracies = Azs;
trainpredictions = cell(1,length(subjects));

% Loop over subjects
for i = 1:length(subjects);
    
    clear EEG; clear X_EEG_TRAIN; clear X_EEG_TEST; clear Y_EEG_TRAIN % clear variables
    
    % Load data
    LOAD_PATH = fullfile('data', ['data_v' num2str(PROCESSED_DATA_VERSION)], ['Subject_', num2str(subjects(i)), '.mat']);
    load(LOAD_PATH);
    
    % Shuffle training data and labels (so initialization is not biased)
    neworder = randperm(length(Y_EEG_TRAIN));
    truelabels = Y_EEG_TRAIN(neworder);
    reordertrain = X_EEG_TRAIN(:,:,neworder);
    
    % Manipulate training data 
    % in this case, take average in time over particular window
    if isempty(channels)
        x = squeeze(mean(reordertrain(:,timewindow,:),2));
    else
        x = squeeze(mean(reordertrain(channels,timewindow,:),2));
    end
    x = x';
    
    % Other possibilities
    %     reordertrain = reordertrain(:,250:500,:);
    % x = reshape(reordertrain,size(reordertrain,1)*size(reordertrain,2),size(reordertrain,3));
    % x = x';
    % possible ways to improve: reduce to only relevant channels?
    
    % Run kmeans
    [idx,c,sumd,d] = kmeans(x,2);
    idx = idx-1; % correct indices to match Y_EEG_TRAIN
    
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

%% Results 

figure; 
plot(Azs); hold on;
plot(accuracies);
legend('Az','accuracy');
title(['Results by Subject for ' classifiername]);
ylim([0 1]);
xlabel('subject'); ylabel('Az');

%% Save and finish

% Save data if requested
if savedata
    % Set up params so this can be recreated
    params.channels = channels;
    params.timewindow = timewindow;
    params.dataversion = PROCESSED_DATA_VERSION;
    % Set savepath
    resultpath = fullfile('results',[classifiername classifiertag '_' num2str(PROCESSED_DATA_VERSION)]);
    save(resultpath,'Azs','trainpredictions','params');
end

disp('done')