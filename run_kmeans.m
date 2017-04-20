%% Initial setup
clear all; clc;
runkmeans = 1;
runsvm = 0;
runlda = 1;
classifiertag = ''; % tag for saving

%% SETTINGS

% saving/plotting
savedata = 1; % will save data to results folder
ploteachsubject = 0; % will plot ROC for every subject

% which data to use
PROCESSED_DATA_VERSION = 0; % which data to load (currently only works with version 0; needs the X_EEG_TRAIN variable to be intact as chanxtimexepoch)
subjects = 1:8; % which subjects to run

% for classifier
epochsize = 50;
channels = [];

%% Load data and build classifier
start = 1; i = 1;
while start < 1000
    timewindow = start:start+epochsize;
    if runkmeans
        [ Azs_kmeans(:,i), trainpredictions_kmeans(:,i), accuracies_kmeans(:,i) ] = classify_kmeans( subjects,PROCESSED_DATA_VERSION,channels,timewindow,ploteachsubject);
    end
    if runsvm
        [ Azs_svm(:,i), trainpredictions_svm(:,i), accuracies_svm(:,i) ] = classify_svm( subjects,PROCESSED_DATA_VERSION,channels,timewindow,ploteachsubject);
    end
    if runlda
        [ Azs_lda(:,i), trainpredictions_lda(:,i), accuracies_lda(:,i) ] = classify_lda( subjects,PROCESSED_DATA_VERSION,channels,timewindow,ploteachsubject);
    end
    start = timewindow(end);
    i = i+1;
end

%% Results

if runkmeans
    figure;
    plot(Azs_kmeans');
    title('Results by Subject for K-Means');
    ylim([0 1]);
    xlabel('timebin'); ylabel('Az');
end

if runsvm
    figure;
    plot(Azs_svm');
    title('Results by Subject for SVM');
    ylim([0 1]);
    xlabel('timebin'); ylabel('Az');
end

if runlda
    figure;
    plot(Azs_lda');
    title('Results by Subject for LDA');
    ylim([0 1]);
    xlabel('timebin'); ylabel('Az');
end

%% Save and finish

% Save data if requested
if savedata
    % Set up params so this can be recreated
    params.channels = channels;
    params.timewindow = timewindow;
    params.dataversion = PROCESSED_DATA_VERSION;
    params.epochsize = epochsize;
    % Set savepath
    if runkmeans
        resultpath = fullfile('results',['kmeans' classifiertag '_' num2str(PROCESSED_DATA_VERSION)]);
        save(resultpath,'Azs_kmeans','trainpredictions_kmeans','params');
    end
    if runsvm
        resultpath = fullfile('results',['svm' classifiertag '_' num2str(PROCESSED_DATA_VERSION)]);
        save(resultpath,'Azs_svm','trainpredictions_svm','params');
    end
    if runlda
        resultpath = fullfile('results',['lda' classifiertag '_' num2str(PROCESSED_DATA_VERSION)]);
        save(resultpath,'Azs_lda','trainpredictions_lda','params');
    end
end

disp('done')

