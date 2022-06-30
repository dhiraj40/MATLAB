warning('off','all');

tic;
[fname,path]=uigetfile('*.*','Browse Image');
image = double(imread([path,fname]))/255;

 image = imresize(image, 0.4);

result = dehaze_fast(image, 0.95, 5);
toc;

figure, imshow(image)
figure, imshow(result)

warning('on','all');