load resnet_filter1

weights = double(permute(weights, [1,2,4,3]));

img_dir = 'figure/resnet_filter/';
colormap gray
img_width = 20;
imtoshow = zeros(img_width*img_width, 64);
for i = 1: 64
    im = weights(:,:,i);
    im = imresize(im, [img_width,img_width], 'bicubic');
    imtoshow(:,i) = im(:);
%     imagesc(weights(:,:,i))
%     axis square
%     axis off
end

img = ml_data2Img_ws(imtoshow, img_width, img_width, 5, 8, 8, 'gray');
imshow(img)
