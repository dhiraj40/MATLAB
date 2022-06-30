function result = modifyImg(I)

%% Apply median filter to smoothen the image
I1 = rgb2gray(I);
K = medfilt2(I1);
figure(45),imshow(K);
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

result = dehaze_fast(rgbResult, 0.95, 5);
toc;

net = denoisingNetwork("dncnn");
result = cnnDenoise(result, net);


end