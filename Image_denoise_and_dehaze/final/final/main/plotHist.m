function [] = plotHist(I)

[redBand, greenBand, blueBand] = imsplit(I);
warning('on','all');
	fontSize = 10;
	
	% Compute and plot the red histogram.
	[countsR, grayLevelsR] = imhist(redBand);
	maxGLValueR = find(countsR > 0, 1, 'last');
	maxCountR = max(countsR);
% 	bar(countsR, 'r');
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
% 	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the green histogram.
	[countsG, grayLevelsG] = imhist(greenBand);
	maxGLValueG = find(countsG > 0, 1, 'last');
	maxCountG = max(countsG);
% 	bar(countsG, 'g', 'BarWidth', 0.95);
% 	grid on;
% 	xlabel('GRAY VALUE');
% 	ylabel('PIXEL');
%     title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the blue histogram.
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
	title('ALL OVER  REGION', 'FontSize', fontSize);


end