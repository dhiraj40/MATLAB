warning('off','all');

tic;
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
image = (imread([pathname,filename]))/255;

image = imresize(image, 0.4);

result = dehaze(image, 1.95, 15);
toc;

figure, imshow(image)
figure, imshow(result)

warning('on','all');