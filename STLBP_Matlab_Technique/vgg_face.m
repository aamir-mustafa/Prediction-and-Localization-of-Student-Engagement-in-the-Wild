face_path='vggfaces/faces';
folder=dir(face_path);
for j=3:size(folder,1)
    person=folder(j).name;
    face_images=dir(strcat(face_path,'/',person));
    segments= zeros(59,20,2622);
    label=zeros(59,1);
    labels_32=load ('labels_32.mat');
    labels_32=labels_32.label;
    count_jj=1;

    for jj=1:10:590
        label(count_jj,1)=labels_32(j-2);
        count_i=1;
        for i= jj:jj+19
            segments(count_jj,count_i,:)= features(:,i);
            count_i=count_i+1;
        end
        j
        count_jj
        
        count_jj=count_jj+1;
    end
    person_name=split(person,'.');
save_name=strcat(person_name{1},'_segments_vgg.mat');
save(save_name,'segments','label')
end