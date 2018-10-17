files=dir('segmented_vgg_label');
final_vgg=zeros(1888,20,2622);
final_label=zeros(1888,1);
for i=3:size(files)
    load(strcat('segmented_vgg_label/',files(i).name))
    final_vgg(1+59*(i-3):59*(i-2),:,:)=segments;
    final_label(1+59*(i-3):59*(i-2),1)=label;
    
end

save 'final_vgg_label_32.mat' final_vgg final_label