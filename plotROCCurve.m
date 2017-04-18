function [Az,accuracy,swaplabels] = plotROCCurve(truelabel,predictedlabel,plotROC,classifiername )


if ~exist('classifiername','var') || isempty(classifiername)
    classifiername = '';
end
if ~exist('plotROC','var') || isempty(plotROC)
    plotROC = 1;
end


% Calculate Az and curve using perfcurve
[falsepos,truepos,~,Az] = perfcurve(truelabel,predictedlabel,1); % finds false and true positives, plus area under the curve

% If Az is less than .5, the labels are reversed
if Az>.5
    swaplabels = 0;
else
    swaplabels = 1;
end

% Calculate accuracy (usually extremely close to Az)
accuracy = sum(truelabel == predictedlabel) / length(truelabel);

% Plot ROC if desired
if plotROC
    figure;
    plot(falsepos,truepos);
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(['Receiver Operator Curve; Az = ' num2str(Az)])
    suptitle(['Model Results for ' classifiername]);
end

end

