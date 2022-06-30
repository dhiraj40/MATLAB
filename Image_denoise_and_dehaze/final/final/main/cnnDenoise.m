
function cnnDenoise = cnnDenoise(I,net)

[noiseR, noiseG, noiseB] = imsplit(I);

denoisedR = denoiseImage(noiseR,net);
denoisedG = denoiseImage(noiseG,net);
denoisedB = denoiseImage(noiseB,net);

cnnDenoise = cat(3,denoisedR,denoisedG,denoisedB);
end






