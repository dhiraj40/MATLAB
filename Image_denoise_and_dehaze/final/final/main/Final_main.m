clc
clear all
close all
tic;
[fname,path]=uigetfile('*.*','Browse Image');
I = imread([path,fname]);
Istrech = imadjust(I,stretchlim(I));
figure(1),imshow(Istrech)
title('Contrast stretched image')


%% Convert RGB image to gray
I1 = rgb2gray(Istrech);
figure(2),imshow(I1,[])
title('RGB to gray (contrast stretched) ')


% ## Gaussian_filter apply

gaussianFilter = fspecial('gaussian',20, 10);
img_filted = imfilter(I1, gaussianFilter,'symmetric');
figure(3)
imshow(img_filted);
title('gaussianFilter Filted Image');

% ##canny edge filter apply 

filted_edges = edge(img_filted, 'Canny');
figure(4);
subplot(121);
imshow(filted_edges);
title('Edges found in filted image');
img_edges = edge(I1, 'Canny');
subplot(122);
imshow(img_edges);


%% Apply median filter to smoothen the image

K = medfilt2(I1);
figure(45),imshow(K)
title('median filter')

%MSE and PSNR measurement

disp('<--------------- Median  filter  ---------------------------->');
disp('Mean Square Error ');
disp(MSE(I1,K));
disp('Peak Signal to Noise Ratio');
disp(PSNR(I1,K));
disp('<--------------------------------------------------------->');


image = I;
hsvImage = rgb2hsv(image);
    
imageHeq = histeq(hsvImage(:,:,3));
hsvResult = hsvImage;
hsvResult(:,:,3) = imageHeq;

rgbResult = hsv2rgb(hsvResult);

result = dehaze(rgbResult, 0.95, 5);
toc;


%title('REMOVE NOISE')
fontSize = 10;
	redBand = I(:, :, 1);
	greenBand = I(:, :, 2);
	blueBand = I(:, :, 3);
	% Display them.
% 	figure
% 	imshow(redBand);
% 	title('Red Band', 'FontSize', fontSize);
% 	figure
% 	imshow(greenBand);
% 	title('Blue Band', 'FontSize', fontSize);
% 	figure
% 	imshow(blueBand);
% 	title('Green Band', 'FontSize', fontSize);
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------- Segmentation ----------------------------------------- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 %result = segmentation(result);


tic;                                                                                                                                                
hsvImage = rgb2hsv(result);
grayImg=result(:,:,3);                                                   
GIm = imcomplement(grayImg); 

HIm = adapthisteq(GIm);    

%Contrast Limited Adaptive Histogram Equalization
figure,
imshow(HIm);
title('HISTOGRAM ADAPTIVE(Gray)');
hsvResult = hsvImage;
hsvResult(:,:,3) = HIm;

rgbResult = hsv2rgb(hsvResult);
figure,
imshow(rgbResult);
title('HISTOGRAM ADAPTIVE(RGB)');


se = strel('ball',8,8);                                                    %Structuring Element
gopen = imopen(HIm,se);                                                    %Morphological Open
godisk = HIm - gopen;                                                      %Remove Optic Disk

medfilt = medfilt2(godisk);                                                %2D Median Filter
background = imopen(medfilt,strel('disk',100));                            %imopen function
I2 = medfilt - background;                                                 %Remove Background
GC = imadjust(I2);                                                         %Image Adjustment
figure(55),
imshow(GC);
title('ADJUST IMAGES');

IM=GC;
IM=double(IM);
[maxX,maxY]=size(IM);
IMM=cat(3,IM,IM);

cc1=8;
cc2=250;

ttFcm=0;

while(ttFcm<10)
    ttFcm=ttFcm+1;
    
    sttFcm=(['ttFcm = ' num2str(ttFcm)]);
   
    
    c1=repmat(cc1,maxX,maxY);
    c2=repmat(cc2,maxX,maxY);
    
    if ttFcm==1 
        test1=c1; test2=c2;
    end
    
    c=cat(3,c1,c2);
    ree=repmat(0.000001,maxX,maxY);
    ree1=cat(3,ree,ree);
    
    distance=IMM-c;
    distance=distance.*distance+ree1;
    
    daoShu=1./distance;
    
    daoShu2=daoShu(:,:,1)+daoShu(:,:,2);
    distance1=distance(:,:,1).*daoShu2;
    u1=1./distance1;
    distance2=distance(:,:,2).*daoShu2;
    u2=1./distance2;
      
    ccc1=sum(sum(u1.*u1.*IM))/sum(sum(u1.*u1));
    ccc2=sum(sum(u2.*u2.*IM))/sum(sum(u2.*u2));
   
    tmpMatrix=[abs(cc1-ccc1)/cc1,abs(cc2-ccc2)/cc2];
    pp=cat(3,u1,u2);
    
    for i=1:maxX
        for j=1:maxY
            if max(pp(i,j,:))==u1(i,j)
                IX2(i,j)=1;
            else
                IX2(i,j)=2;
            end
        end
    end
    
    if max(tmpMatrix)<0.0001
        break;
    else
        cc1=ccc1;
        cc2=ccc2;
    end

    for i=1:maxX
        for j=1:maxY
            if IX2(i,j)==2
                IMMM(i,j)=254;
            else
                IMMM(i,j)=8;
            end
        end
    end
    
    background=imopen(IMMM,strel('disk',45));
    I4=IMMM-background;
    I4=bwareaopen(I4,30);
    figure(55),
imshow(I4);
title('ADJUST IMAGES');
 
for i=1:maxX
    for j=1:maxY
        if IX2(i,j)==2
            IMMM(i,j)=200;
        else
            IMMM(i,j)=1;
        end
    end
end 

ffcm1=(['The 1st Cluster = ' num2str(ccc1)]);
ffcm2=(['The 2nd Cluster = ' num2str(ccc2)]);
[m,n]=size(I4);
Tn=0;
Tp=0;
Fp=0;
Fn=0;

for i=1:m
    for j=1:n
        if I4(i,j)==0 && I(i,j)==0 
            Tn=Tn+1;
        elseif I4(i,j)==1 && I(i,j)==1
            Tp=Tp+1;
        elseif  I4(i,j)==1 && I(i,j)==0
            Fp=Fp+1;
        elseif  I4(i,j)==0 && I(i,j)==1
            Fn=Fn+1;
        end
    end
end

end
aucc=(Tp+Tn)/(Tp+Tn+Fp+Fn);                                                %Accuracy                                                 
sensitivity=Tp/(Tp+Fn);                                                    %True Positive Rate
specificity=Tn/(Tn+Fp);                                                    %True Negative Rate
fpr=1-specificity;                                                         %False Positive Rate
ppv=Tp/(Tp+Fp);                                                            %Positive Predictive Value
disp('True Positive = ');
disp(Tp);
disp('True Negative = ');
disp(Tn);
disp('False Positive = ');
disp(Fp);
disp('False Negative = ');
disp(Fn);
disp('False Positive Rate = ');
disp(fpr);
disp('Sensitivity = ');
disp(sensitivity);
disp('Specificity = ');
disp(specificity);
disp('Accuracy = ');
disp(aucc);
disp('Positive Predictive Value = ');
disp(ppv);



fontSize = 10;
	redBand = I(:, :, 1);
	greenBand = I(:, :, 2);
	blueBand = I(:, :, 3);
% 	% Display them.
% 	figure
% 	imshow(redBand);
% 	title('CONTRAST 1', 'FontSize', fontSize);
% 	figure
% 	imshow(greenBand);
% 	title('CONTRAST 2', 'FontSize', fontSize);
% 	figure
% 	imshow(blueBand);
% 	title('CONTRAST 3', 'FontSize', fontSize);
%     
% tic;
% I = imresize(I,[200,200]);
% Img = double(I(:,:,1));
% epsilon = 1;
% 
% num_it =1600;
% rad = 9;
% alpha = 0.003;% coefficient of the length term
% mask_init = zeros(size(Img(:,:,1)));
% mask_init(53:77,56:70) = 1;
% seg = local_AC_MS(Img,mask_init,rad,alpha,num_it,epsilon);

net = denoisingNetwork("dncnn");
result = cnnDenoise(result, net);

figure,
imshow(I);
title("Original Image")
figure,
imshow(result);
title("Denoised Image")
HighIm = EDSR_2xSuperResolution(result);
figure,
imshow(HighIm);
title("High Resolution Denoised Image")

%MSE and PSNR measurement
image = I;
mse = (MSE(image(:,:,1),result(:,:,1)) + MSE(image(:,:,2),result(:,:,2)) + MSE(image(:,:,3),result(:,:,3)))/3;
psnr = 10*(log(255*255/mse) / log(10));
disp('<--------------- Proposed  Method  ---------------------------->');
disp('Mean Square Error ');
disp(mse);
disp('Peak Signal to Noise Ratio');
disp(psnr);
disp('<--------------------------------------------------------->');



[redBand, greenBand, blueBand] = imsplit(image);

warning('on','all');
	fontSize = 10;
	
	% Compute and plot the red histogram.
	% hR = figure;
	[countsR, grayLevelsR] = imhist(redBand);
	maxGLValueR = find(countsR > 0, 1, 'last');
	maxCountR = max(countsR);
% 	bar(countsR, 'r');
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
% 	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the green histogram.
	% hG = figure
	[countsG, grayLevelsG] = imhist(greenBand);
	maxGLValueG = find(countsG > 0, 1, 'last');
	maxCountG = max(countsG);
% 	bar(countsG, 'g', 'BarWidth', 0.95);
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
%     title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the blue histogram.
	% hB = figure
	[countsB, grayLevelsB] = imhist(blueBand);
	maxGLValueB = find(countsB > 0, 1, 'last');
	maxCountB = max(countsB);
% 	bar(countsB, 'b');
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
% 	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Set all axes to be the same width and height.
	% This makes it easier to compare them.
	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]);
% 	if eightBit
% 		maxGL = 255;
% 	end
	maxCount = max([maxCountR,  maxCountG, maxCountB]);
% 	axis([hR hG hB], [0 maxGL 0 maxCount]);
	
% 	Plot all 3 histograms in one plot.
	figure
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2);
	grid on;
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	hold on;
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2);
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2);
	title('Original Image', 'FontSize', fontSize);

% <--------------------------------------------------->

[redBand, greenBand, blueBand] = imsplit(result);

warning('on','all');
	fontSize = 10;
	
	% Compute and plot the red histogram.
	% hR = figure;
	[countsR, grayLevelsR] = imhist(redBand);
	maxGLValueR = find(countsR > 0, 1, 'last');
	maxCountR = max(countsR);
% 	bar(countsR, 'r');
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
% 	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the green histogram.
	% hG = figure
	[countsG, grayLevelsG] = imhist(greenBand);
	maxGLValueG = find(countsG > 0, 1, 'last');
	maxCountG = max(countsG);
% 	bar(countsG, 'g', 'BarWidth', 0.95);
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
%     title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the blue histogram.
	% hB = figure
	[countsB, grayLevelsB] = imhist(blueBand);
	maxGLValueB = find(countsB > 0, 1, 'last');
	maxCountB = max(countsB);
% 	bar(countsB, 'b');
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
% 	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Set all axes to be the same width and height.
	% This makes it easier to compare them.
	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]);
% 	if eightBit
% 		maxGL = 255;
% 	end
	maxCount = max([maxCountR,  maxCountG, maxCountB]);
% 	axis([hR hG hB], [0 maxGL 0 maxCount]);
	
% 	Plot all 3 histograms in one plot.
	figure
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2);
	grid on;
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	hold on;
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2);
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2);
	title('Modified Image', 'FontSize', fontSize);


