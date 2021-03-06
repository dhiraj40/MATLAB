
function [ net ] = CreateNet( conf_file )
%% Constructor for net struct , loads the configuration from conf_file

     net = initNetDefaults;
	 
     txt=fileread(conf_file);
     eval(txt);
     
     net.properties.numLayers  = length(net.layers);
     net.properties.version    = 2.3;

     conf_dirStruct = dir(conf_file); conf_dirStruct.name=conf_file;
     net.properties.sources{1}=[dir('./*.m') ; dir('./Util/*.m') ; conf_dirStruct];
     for i=1:length(net.properties.sources{1})
         net.properties.sources{1}(i).data = fileread(net.properties.sources{1}(i).name);
     end
     
     net.runInfoParam.iter=0;
     net.runInfoParam.samplesLearned=0;
     net.runInfoParam.maxsucessRate=0;
     net.runInfoParam.noImprovementCount=0;
     net.runInfoParam.minLoss=Inf;
     net.runInfoParam.improvementRefLoss=inf;
     
 
     net = initNetWeight(net);

     net.properties.numOutputs = prod(net.layers{end}.properties.sizeOut);

     net.runInfoParam.endSeed = rng;

     printNetwork(net);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ net ] = initNetDefaults( net )

net.hyperParam.trainLoopCount=1000;%on how many samples to train before evaluating the network
net.hyperParam.testImageNum=2000;
net.hyperParam.batchNum = 1; % on how many samples to average weights update
net.hyperParam.ni_initial    = 0.05;% ni to start training process
net.hyperParam.ni_final = 0.00001;% final ni to stop the training process
net.hyperParam.noImprovementTh=50; % if after noImprovementTh there is no improvement , reduce ni
net.hyperParam.momentum=0;  
net.hyperParam.constInitWeight=nan; %Use nan to set initial weight to random. Any other value to fixed 
net.hyperParam.lambda=0; %L2 regularization factor, set 0 for none. Above 0.01 not recommended
net.hyperParam.testOnData=0; % to perform testing after each epoc on the data inputs or test inputs
net.hyperParam.addBackround=0; % random background can be added to samples before passing to net in order to improve noise resistance.
net.hyperParam.testOnNull=0;% Training on non data images without any feature to detect 
net.properties.skipLastLayerErrorCalc=1; % the input layer does not need errors hence calculation can be skipped     

%%%%%%%%%%%%%% Augmentation %%%%%%%%%%%%%%
net.hyperParam.augmentImage=0; % set to 0 for no augmentation
net.hyperParam.augmentParams.noiseVar=0.02;
net.hyperParam.augmentParams.maxAngle=45/3;
net.hyperParam.augmentParams.maxScaleFactor=1.1;
net.hyperParam.augmentParams.minScaleFactor=1/1.5;
net.hyperParam.augmentParams.maxStride=4;

net.hyperParam.augmentParams.maxSigma=2;%for gauss filter smoothing
net.hyperParam.augmentParams.imageComplement=0;% will reverse black/white of the image

net.hyperParam.augmentParams.medianFilt=0; %between 0 and one - if this value is 0.75 it will zero all 75% lower points. 0 will mean no point is changed, 1 will keep the higest point only 

%%%%%%%%%%%%%% Centralize image before passing to net? %%%%%%%%%%%%%%

net.hyperParam.centralizeImage=0;
net.hyperParam.cropImage=0;
net.hyperParam.flipImage=0;           % fill randomly flip the input hor/vert before passing to the network. Improves learning in some instances
net.hyperParam.useRandomPatch=0;
net.hyperParam.testNumPatches=1;      % on how many patches from a single image to perform testing. network is evaluated on several patches and result is averaged over all patches.
net.hyperParam.selevtivePatchVarTh=0; %in order to drop patches that their variance is less then th
net.hyperParam.testOnMiddlePatchOnly=0; %will test on the middle patch only
net.hyperParam.normalizeNetworkInput=1; %will normalize every input to net to be with var=1, mean 0
net.hyperParam.randomizeTrainingSamples=1; % randomize the samples selected from dataset during training 



%%%%%%%%%%%%%% Run info - parameters that change every iteration %%%%%%%%%%%%%%
net.runInfoParam.storeMinLossNet= 0; % this enables the trainer to store also the net with the lowest loss and max success rate found (in addition to the latest one)
net.runInfoParam.verifyBP       = 1; % can perform pre-train back-propagation verification. Useful to detect faults in the application

%%%%%%%%%%%%%% types %%%%%%%%%%%%%%
net.layers={};
net.types.input='input';
net.types.fc='fc';
net.types.conv='conv';
net.types.batchNorm='batchNorm';
net.types.output='output';
net.types.softmax='softmax';
net.types.reshape='reshape';
end

