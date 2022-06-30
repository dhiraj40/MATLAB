
%Program for Peak Signal to Noise Ratio Calculation

function PSNRV = PSNR(noiseImg, filteredImg)

noiseImg = double(noiseImg);
filteredImg = double(filteredImg);

[M, N] = size(noiseImg);
error = noiseImg - filteredImg;
MSE = sum(sum(error .* error)) / (M * N);

if(MSE > 0)
    PSNRV = 10*log(255*255/MSE) / log(10);
else
    PSNRV = 99;
end
