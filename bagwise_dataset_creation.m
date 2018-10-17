clear
clc
load('Normalized_relabeled_lbptop_20_segments_with_labels_taking_max1_150_per_video.mat');
count=1;

lbptop_feats_bagwise=zeros(53,150,177);
labels_bagwise= zeros(53,1);
%combined_lbptop_20_segments_54--- 7950 x 177
%labels---- 7950 x 1
for i =1:150:7801
    count
    a=labels(i:i+149);
    labels_bagwise(count,1)=mean(a);
    lbptop_feats_bagwise(count,:,:)=combined_lbptop_20_segments_53(i:i+149,:);
    
    count=count+1;
end

save Norm_bagwise_lbptop_feats_53_with_relabeled_KMeans_max1.mat labels_bagwise lbptop_feats_bagwise

