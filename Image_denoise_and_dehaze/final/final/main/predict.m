clc
clear all
close all
tic;
[fname,path]=uigetfile('*.*','Browse Image');
I = double(imread([path,fname]))/255;


result = modifyImg(I);
figure,
imshow(I);
title("Original Image")
figure,
imshow(result);
title("Denoised Image")

plotHist(I);
plotHist(result);

image = I;
%MSE and PSNR measurement

mse = (MSE(image(:,:,1),result(:,:,1)) + MSE(image(:,:,2),result(:,:,2)) + MSE(image(:,:,3),result(:,:,3)))/3;
psnr = 10*(log(255*255/mse) / log(10));
disp('<--------------- Proposed  Method  ---------------------------->');
disp('Mean Square Error ');
disp(mse);
disp('Peak Signal to Noise Ratio');
disp(psnr);
disp('<--------------------------------------------------------->');

