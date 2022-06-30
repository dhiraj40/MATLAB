function MSE = MSE(noiseImg, filteredImg)

noiseImg = double(noiseImg);
filteredImg = double(filteredImg);

[M N] = size(noiseImg);
error = noiseImg - filteredImg;
MSE = sum(sum(error .* error)) / (M * N);
end