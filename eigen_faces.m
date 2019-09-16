%% Author: Junyu Chen

clear all; close all;

%% read all training images
imdata = [];

% go to every folder and get information
for folder_index = 1:40
    folder_index_str = num2str(folder_index);
    folder_location = strcat('orl_faces/Train/s',folder_index_str);
    cd(folder_location);
    imdata_temp = zeros(9, 112*92);
    for file_index = 1:9
        imdata_temp_1 = imread(strcat(num2str(file_index),'.pgm'));
        imdata_temp(file_index,:) = reshape(imdata_temp_1, [1, 112*92]);
    end
    imdata = [imdata; imdata_temp];
    cd('../../../');
end
imdata = imdata';

%% Center dataset
mean_img = mean(imdata,2);
for i = 1:size(imdata,2)
    imdata(:,i) = imdata(:,i)-mean_img;
end
figure; imshow(reshape(mean_img,[112,92]),[])

%% Covariance Matrix
cor_mat = imdata'*imdata;

%% Eigen vector
[V,D] = eig(cor_mat);
egVal = diag(D);
%% order by largest eigenvalues
egVal = egVal(end:-1:1);
V = V(:,end:-1:1);
plot(egVal);xlabel('indexes'); ylabel('eigen values'); title('eigen values')
%% Principal Component Analysis
% normalize eigenvector by the eigenvalue
for i = 1:size(V,2)
    V(:,i) = V(:,i)./sqrt(egVal(i));
end
eigFaces = imdata*V;
%eigFaces = eigFaces*diag(1./sqrt(egVal(:)));
%for i = 1:size(eigFaces,2)
%    eigFaces(:,i) = eigFaces(:,i)./sqrt(egVal(i));
%end

%% Plot eigenfaces
figure; imshow(reshape(eigFaces(:,1),[112,92]),[])
title('the first eigenface')
%% project eigenfaces
figure;
i = 1;
for numOfEig = 10:100:360
    eigFaces_sub = eigFaces(:,1:numOfEig);
    wt=imdata(:,1)'*eigFaces_sub; % weighting
    vi=reshape(eigFaces_sub*wt',112,92); %projection
    subplot(2,2,i);imshow(vi,[]); title(['recon. faces with ',num2str(numOfEig),' eigen faces'])
    i = i+1;
end

%% eigenface detection-training image
proj_train=zeros(112*92,40,10);
all_faces=zeros(112*92,40,10);
for ni=1:40
    for kimg=1:9
        filename=sprintf('orl_faces/Train/s%i/%i.pgm',ni,kimg);
        images{ni,kimg}=imread(filename);
        one_face=images{ni,kimg};% load one face
        % vectorize image
        all_faces(:,ni,kimg)=reshape(one_face,112*92,1);
        % remove mean
        one_face=double(all_faces(:,ni,kimg))-mean_img;
        % calculate weights
        cni=one_face'*eigFaces;
        % project onto eigenvector
        proj_train(:,ni,kimg)=eigFaces*cni';
    end
end

%% test images
figure;
for test_category=1:10
    test_filename=sprintf('orl_faces/Test/s%i/%i.pgm',test_category,10);
    test_img=imread(test_filename);
    orgImg=test_img;
    test_img=reshape(test_img,112*92,1);
    test_img=double(test_img)-mean_img;
    test_wt=test_img'*eigFaces;
    test_img_proj=eigFaces*test_wt';

    dist=zeros(40,9);
    for ni=1:40
        for kimg=1:9
            % calculate squared dist. in projection domain
            dist(ni,kimg)=sum((test_img_proj-proj_train(:,ni,kimg)).^2);
        end
    end
    [min_val,idx]=min(dist(:));
    [class_pred,col]=ind2sub(size(dist),idx);
    sprintf('Test category and predicted class are %i, %i',test_category, class_pred)
    subplot(211)
    imshow(orgImg)
    subplot(212)
    imshow(reshape(all_faces(:,class_pred,col),112,92),[])
    pause

end