clear all;
close all;
%A variable to store the input for the imput neurons
input1=zeros(40,10,100);
ypat=zeros(40,10); %variable to store the global average for input to emotional neurons.
filePattern = 'D:\Internship IIIT Hyderabad\att_faces\orl_faces';
fileList1=dir(filePattern);
%for i=1
%-2 is done because the first two are automatically created so our interest
%data id from 3 onwards.
for i=1:length(fileList1)-2
    filename=fileList1(i+2).name; %retrive each folder of data (like s1, s2,etc)
    fileList = dir(strcat(filePattern,'\',filename)); %list all the pgm files
    for k = 1:length(fileList)-2
    %for k=1
        thisFileName = fileList(k+2).name; %retrive the pgm files one by one
        thisImage = imread(fullfile(strcat(filePattern,'\',filename),thisFileName)); %read the selected pgm file
        thisImage=imresize(thisImage,[100 100]); %resizing of the image
        thisImage=thisImage./max(thisImage(:));
        j=1;
            for m=0:9
                for n=0:9
                    a=thisImage(m*10+1:(m+1)*10,n*10+1:(n+1)*10); %segmenting the image to 10X10 segments each
                     input1(i,k,j)=mean(mean(a)); %finding the mean of each segment
                     j=j+1;
                end
            end
            
            ypat(i,k)=mean(mean((thisImage(:)))); %Input to emotional neuron
    end
end


%ypat=ypat./max(ypat(:));
%input1=input1./max(input1(:));


%Preparing the training and testing data 
%1. 50% of data
training1=input1(1:40,1:5,:);
testing1=input1(1:40,6:10,:);

%2. 40% of data for training
training2=input1(1:40,1:4,:);
testing2=input1(1:40,5:10,:);

%3. 30% of data for training
training3=input1(1:40,1:3,:);
testing3=input1(1:40,4:10,:);


save('dataset1','training1','testing1');
save('dataset2','training2','testing2');
save('dataset3','training3','testing3');


%1.label for data
label1=zeros(400,40);
k=1;
for i=1:200
    label1(i,k)=1;
    if(mod(i,5)==0)
        k=k+1;
    end
end
k=1;
for i=201:400
    label1(i,k)=1;
    if(mod(i,5)==0)
        k=k+1;
    end
end

%1. label for dataset1
label1train=label1(1:200,:);
label1test=label1(201:400,:);
save('label1','label1train','label1test');

label2=zeros(400,40);
k=1;
for i=1:160
    label2(i,k)=1;
    if(mod(i,4)==0)
        k=k+1;
    end
end
k=1;
count=0;
for i=161:400
    label2(i,k)=1;
    count=count+1;
    if(count==6)
        k=k+1;
        count=0;
    end
end

%2.label for dataset2
label2train=label2(1:160,:);
label2test=label2(161:400,:);
save('label2','label2train','label2test');

label3=zeros(400,40);
k=1;
for i=1:120
    label3(i,k)=1;
    if(mod(i,3)==0)
        k=k+1;
    end
end
k=1;
count=0;
for i=121:400
    label3(i,k)=1;
    count=count+1;
    if(count==7)
        k=k+1;
        count=0;
    end
end

%3. label for dataset3

label3train=label3(1:120,:);
label3test=label3(121:400,:);
save('label3','label3train','label3test');


%saving input to emotional neurons
ypattrain1=ypat(:,1:5);
ypattest1=ypat(:,6:10);
save('ypat1','ypattrain1','ypattest1');

ypattrain2=ypat(:,1:4);
ypattest2=ypat(:,5:10);
save('ypat2','ypattrain2','ypattest2');

ypattrain3=ypat(:,1:3);
ypattest3=ypat(:,4:10);
save('ypat3','ypattrain3','ypattest3');
            