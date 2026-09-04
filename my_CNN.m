% Helper function
function downloadExampleFoodImagesData(url, dataDir)
% Download the Example Food Image data set, containing 978 images of
% different types of food split into 9 classes.

% Copyright 2019 The MathWorks, Inc.

fileName = "ExampleFoodImageDataset.zip";
fileFullPath = fullfile(dataDir, fileName);

% Download the .zip file into a temporary directory.
if ~exist(fileFullPath, "file")
    fprintf("Downloading MathWorks Example Food Image dataset...\n");
    fprintf("This can take several minutes to download...\n");
    websave(fileFullPath, url);
    fprintf("Download finished...\n");
else
    fprintf("Skipping download, file already exists...\n");
end

% Unzip the file.
%
% Check if the file has already been unzipped by checking for the presence
% of one of the class directories.
exampleFolderFullPath = fullfile(dataDir, "pizza");
if ~exist(exampleFolderFullPath, "dir")
    fprintf("Unzipping file...\n");
    unzip(fileFullPath, dataDir);
    fprintf("Unzipping finished...\n");
else
    fprintf("Skipping unzipping, file already unzipped...\n");
end
fprintf("Done.\n");

end


% Download Data Set

dataDir = fullfile(tempdir,"ExampleFoodImageDataset");
url = "https://www.mathworks.com/supportfiles/nnet/data/ExampleFoodImageDataset.zip";

if ~exist(dataDir,"dir")
    mkdir(dataDir);
end

downloadExampleFoodImagesData(url,dataDir);

% Train Network to Classify Food Images

imds = imageDatastore(dataDir, ...
    IncludeSubfolders=true,LabelSource="foldernames");

aug = imageDataAugmenter(RandXReflection=true, ...
    RandYReflection=true, ...
    RandXScale=[0.8 1.2], ...
    RandYScale=[0.8 1.2], ...
    RandRotation=[-30 30],...
    RandXTranslation=[-20 20],...
    RandYTranslation=[-20 20]);

trainingFraction = 0.65;
[trainImds,valImds] = splitEachLabel(imds,trainingFraction);

augImdsTrain = augmentedImageDatastore([227 227],trainImds, ...
    DataAugmentation=aug);
augImdsVal = augmentedImageDatastore([227 227],valImds);
classNames = categories(trainImds.Labels);
numClasses = numel(classNames);

% --------------------------- KEY -----------------------------------------
net = dlnetwork;
tempNet = [
    imageInputLayer([227 227 3],"Name","imageinput")
    
    % Block 1
    convolution2dLayer([5 5],32,"Name","conv_1","Padding","same","WeightL2Factor",0.001)
    batchNormalizationLayer("Name","bn_1")
    reluLayer("Name","relu")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Stride",[2 2])
    
    % Block 2
    convolution2dLayer([5 5],64,"Name","conv_2","Padding","same","WeightL2Factor",0.001)
    batchNormalizationLayer("Name","bn_2")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Stride",[2 2])
    
    % Block 3
    convolution2dLayer([3 3],128,"Name","conv_3","Padding","same","WeightL2Factor",0.001)
    batchNormalizationLayer("Name","bn_3")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Stride",[2 2])

    % Block 4
    convolution2dLayer([3 3],256,"Name","conv_4","Padding","same","WeightL2Factor",0.001)
    batchNormalizationLayer("Name","bn_4")
    reluLayer("Name","relu_3")
   
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(9,"Name","fc_2")
    softmaxLayer("Name","softmax")];

net = addLayers(net,tempNet);

clear tempNet;

net = initialize(net);
% --------------------------- KEY -----------------------------------------

opts = trainingOptions("adam", InitialLearnRate=1e-4, ...
    MaxEpochs=50, ValidationData=augImdsVal, Verbose=false,...
    Plots="training-progress", MiniBatchSize=64, Metrics="accuracy");

rng default

net = trainnet(augImdsTrain,net,"crossentropy",opts);
save("final_net.mat", "net");

expName = "exp20"; 

% Classify Validation Data

figure();
scores = minibatchpredict(net,augImdsVal);
YPred = scores2label(scores,classNames);

confusionchart(valImds.Labels,YPred,ColumnSummary="column-normalized")
saveas(gcf, expName + "_confusion.png")

% Compute Activations for Softmax Layer Only
softmaxLayerName = "softmax";
softmaxActivations = minibatchpredict(net,augImdsVal,Outputs=softmaxLayerName);

numValObservations = augImdsVal.numobservations;
softmaxActivations = reshape(softmaxActivations,numValObservations,[]);

% Ambiguity of Classifications

[R,RI] = maxk(softmaxActivations,2,2);
ambiguity = R(:,2)./R(:,1);
[ambiguity,ambiguityIdx] = sort(ambiguity,"descend");

classList = unique(valImds.Labels);
top10Idx = ambiguityIdx(1:10);
top10Ambiguity = ambiguity(1:10);
mostLikely = classList(RI(ambiguityIdx,1));
secondLikely = classList(RI(ambiguityIdx,2));
table(top10Idx,top10Ambiguity,mostLikely(1:10),secondLikely(1:10),valImds.Labels(ambiguityIdx(1:10)),...
    VariableNames=["Image #","Ambiguity","Likeliest","Second","True Class"])

% Compute 2-D Representation Using t-SNE (Softmax Only)

rng default
softmaxtsne = tsne(softmaxActivations);

% Explore Observations in t-SNE Plot

numClasses = length(classList);
colors = lines(numClasses);
h = figure;
gscatter(softmaxtsne(:,1),softmaxtsne(:,2),valImds.Labels,colors);

l = legend;
l.Interpreter = "none";
l.Location = "bestoutside";
% saveas(gcf, expName + "_softmax_embedding.png")

