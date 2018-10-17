folders=dir('oulucasia\A_Original Video\Original Video\only_videos');
folders=folders(3:end);
final=zeros(1,20,112,112,3);
count=1;
for i=1%:length(folders)
    folder_name=folders(i).name;
    name=strcat('oulucasia\A_Original Video\Original Video\only_videos\', folder_name, '\*.bmp');
    images=dir(name);
    
    no_of_images=length(images);
    
    if (no_of_images<20)
        deficit=20-no_of_images;
        segment=zeros(20,112,112,3);
    for j=1:no_of_images
        
        frame=strcat('oulucasia\A_Original Video\Original Video\only_videos\',folder_name,'\',images(j).name);
        a=imread(frame); %112 x 112 x 3
        segment(j,:,:,:)=a(:,:,:);
     end % j
    for d=1:deficit
        segment(no_of_images+d,:, :, :)= a(:,:,:);S
    end %d
    end %if
    for f=1:5
        final= [final; segment]S
    end %f (this is for repeating five times when no_of_images<20)
    
end %end for i (no_of_images<20)
