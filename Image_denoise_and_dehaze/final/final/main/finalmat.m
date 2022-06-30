clc;
clear all;
close all;

A = 'REAL IMAGES';
b = 'FAKE IMAGES';



srcFiles = dir('C:\Users\ABN_01\Desktop\BIKASH_2020\Bikas\matlab_2021-2022\fake or real images_cnn_final_code\real or fake\final\final\TTD\*.jpg'); 
for i = 1 : length(srcFiles)
   % filename = strcat('E:\2019_dip code\melo\all\',srcFiles(i).name);
   filename = strcat('C:\Users\ABN_01\Desktop\BIKASH_2020\Bikas\matlab_2021-2022\fake or real images_cnn_final_code\real or fake\final\final\TTD\',num2str(i),'.jpg');
    I = imread(filename);
%    I1 = rgb2gray(I);
figure, imshow(I); title('REAL OR FAKE IMAGES');
I = imresize(I,[200,200]);
%figure, imshow(I);
%imhist(image)
%title('input hist');

%ENHANCEMENT
%DILATION
signal1 = feature_ext(I);
[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);

g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast(i,1) = stats.Contrast;
Correlation(i,1) = stats.Correlation;
Energy(i,1) = stats.Energy;
Homogeneity(i,1) = stats.Homogeneity;
Mean(i,1) = mean2(G);
Standard_Deviation(i,1) = std2(G);
Entropy(i,1) = entropy(G);
RMS(i,1) = mean2(rms(G));
%Skewness = skewness(img)
Variance(i,1) = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness(i,1) = 1-(1/(1+a));
Kurtosis(i,1) = kurtosis(double(G(:)));
Skewness(i,1) = skewness(double(G(:)));

meas = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness];
label = {A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;A;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b;b};
%label = {l};
end

save Trainset.mat meas label
