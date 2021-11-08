clc;
clear all;
close all;
warning off;
g=alexnet; %pre-trained AI network
layers=g.Layers;
layers(23)=fullyConnectedLayer(3);
layers(25)=classificationLayer;
allImages=imageDatastore('datasets','IncludeSubfolders',true, 'LabelSource','foldernames');
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
myNet=trainNetwork(allImages,layers,opts);
save myNet3;