clc 
clear

final_combined_lbptop_150_seg_per_video=zeros(53*150, 177);
final_labels_150_seg_per_video= zeros(53*150,1);
%resized_lbptop_person=zeros(150,177);
%resized_label_segment=zeros(150,1);

resized_lbptop_person= lbp_top_person(1:150,:);
resized_label_segment= label_segment(1:150,:);

final_combined_lbptop_150_seg_per_video(7801:7950,:)=resized_lbptop_person;

final_combined_lbptop_150_seg_per_video(7651:7800,:)=new;
%final_labels_150_seg_per_video(1:150,:)=resized_label_segment;

new=zeros(1,177);
for i=1:55
    
    new=[new;lbp_top_person(i,:)];
    
    new=[new;lbp_top_person(i,:)];
    
    
end
new=new(2:end,:);
new=[new; lbp_top_person(56:end,:)];
new=new(1:end-8,:);

for i=1:53
   
    final_labels_150_seg_per_video((i-1)*150+1:(i-1)*150+150)=labels(i);
end







