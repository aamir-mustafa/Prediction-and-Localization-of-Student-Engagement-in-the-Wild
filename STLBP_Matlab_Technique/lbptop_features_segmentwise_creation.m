clear
path= 'C:\Users\sony\Desktop\Internship at IITR\amir\STLBP_Matlab\lbptop_features_segmentwise';
cd(path)
mat_files=dir(path);
lbptop_features_segmentwise=zeros(1888,177);
for i=3:34
    file=load(mat_files(i).name);
    file=file.lbp_top_person;
    lbp=transpose(file);
    lbptop_features_segmentwise(1+59*(i-3):59*(i-2),:)=lbp(:,:);
end
aa=load('C:\Users\sony\Desktop\Internship at IITR\amir\final_vgg_label_32.mat');
label=aa.final_label;
save lbptop_features_segmentwise_with_labels.mat lbptop_features_segmentwise label