clear;
load('label1.mat');
load('label2.mat');
load('label3.mat');

load('dataset1.mat');
load('dataset2.mat');
load('dataset3.mat');

load('ypat1.mat');
load('ypat2.mat');
load('ypat3.mat');


%% For Dataset1
% Training
iteration=0;
a=0;
nh=60;
in=100;
op=40;
e=0.002;
err=1;
learning_rate=0.007;
momentum=0.21;
acf=1;
ccf=0;
error=zeros(200,1);
yhold=zeros(nh,1);
yhnew=zeros(nh,1);
yjold=zeros(1,op);
yjnew=zeros(1,op);
wih= -0.30 + (0.30+0.30)*rand(100,nh);
whj=-0.30 + (0.30+0.30)*rand(nh,op);
whb=-0.30+0.60*rand(1,nh);
whm=-0.30+0.60*rand(1,nh);
wjb=-0.30+0.60*rand(1,op);
wjm=-0.30+0.60*rand(1,op);
deltah=zeros(1,nh);
deltaj=zeros(1,op);
deltawhj=zeros(nh,op);
deltawih=zeros(100,nh);
deltawjb=zeros(1,op);
deltawjm=zeros(1,op);
deltawhb=zeros(1,nh);
deltawhm=zeros(1,nh);
epoch=0;
%while(err>=0.002)
while(epoch<=10)
    a=0;
    sumypat=0;
    for i=1:40
        for j=1:5
         yhold=yhnew;
         yjold=yjnew;
         
         %for input to hidden layer
            for h=1:nh
                sum=0;
                for k=1:100
                    sum=sum+(wih(k,h).*training1(i,j,k));
                end
                %Adding bias
                sum=sum+whb(1,h)*1;
                
                %Adding emotional neuron
                sum=sum+whm(1,h)*ypattrain1(i,j);
                
                yhnew(h,1)=1/(1+exp(-sum));
            end
            
            %For hidden to outer layer
              for z=1:op
                sum=0;
                for h=1:nh
                    sum=sum+(whj(h,z).*yhnew(h,1));
                end
                
                %Adding bias
                sum=sum+wjb(1,z)*1;
                
                %Adding emotional neuron
                sum=sum+whm(1,z)*ypattrain1(i,j);
                
                yjnew(1,z)=1/(1+exp(-sum));
              end
              a=a+1;
              temp=(label1train(a,:)-yjnew(1,:)).*(label1train(a,:)-yjnew(1,:));
              sum=0;
              for z=1:op
                sum=sum+temp(1,z);
              end
              error(a,1)=sum;
              
              %propagating error from output to hidden layer.
              deltaj(1,1:op)=yjnew(1,:).*(1-yjnew(1,:)).*(label1train(a,:)-yjnew(1,:));
              
              %Calculating weight change between output and hidden layer.
              
              for h=1:nh
                  for z=1:op
                      %This is for conventional neurons
                      deltawhj(h,z)=(learning_rate*deltaj(1,z)*yhnew(h,1) + momentum*deltawhj(h,z))+ (acf*deltaj(1,z)*ypattrain1(i,j) + ccf*deltawhj(h,z));
                  end
              end
              %this is for bias
              deltawjb(1,:)=(learning_rate.*deltaj(1,:)*1 + momentum.*deltawjb(1,:)) + (acf*deltaj(1,:)*ypattrain1(i,j) + ccf*deltawjb(1,:));
              %this is for emotional neurons.
              deltawjm(1,:)=(acf*deltaj(1,:)*ypattrain1(i,j) + ccf*deltawjm(1,:));
              
              
              %Propagating error from hidden to input layer
               for h=1:60
                   deltah(1,h)=yhnew(h,1)*(1-yhnew(h,1))*(whj(h,:)*(deltaj(1,:)'));
               end
                      
              %Calculating weight change between hidden and input layer.
              
              for k=1:100
                  for h=1:nh
                      %This is for conventional neurons
                      deltawih(k,h)=(learning_rate*deltah(1,h)*training1(i,j,k) + momentum*deltawih(k,h))+ (acf*deltah(1,h)*ypattrain1(i,j) + ccf*deltawih(i,h));
                  end
              end
              %this is for bias
              deltawhb(1,:)=(learning_rate.*deltah(1,:)*1 + momentum.*deltawhb(1,:)) + (acf*deltah(1,:)*ypattrain1(i,j) + ccf*deltawhb(1,:));
              %this is for emotional neurons.
              deltawhm(1,:)=(acf*deltah(1,:)*ypattrain1(i,j) + ccf*deltawhm(1,:));
              
              
              %Changing the weights for the next iteration
              whj(1:nh,1:op)=whj(1:nh,1:op)+deltawhj(1:nh,1:op);
              wjb(1,:)=wjb(1,:)+deltawjb(1,:);
              wjm(1,:)=wjm(1,:)+deltawjm(1,:);
              
              wih(1:100,1:nh)=wih(1:100,1:nh)+deltawih(1:100,1:nh);
              whb(1,:)=whb(1,:)+deltawhb(1,:);
              whm(1,:)=whm(1,:)+deltawhm(1,:);
              
              iteration=iteration+1;
              sumypat=sumypat+(ypattrain1(i,j));

        end
    end
    epoch=epoch+1;
    sum=0;
        for z=1:200
            sum=sum+error(z,1);
        end
        err=sum/200;
        err=err/40;
        
       er(epoch,1)=err;
        
        yavpat=(sumypat/200);
        
        acf=yavpat+err;
         if(epoch==1)
             acfo=acf;
        end
        
        if(epoch>1)
            ccf=acfo-acf;
        end
        pacf(epoch,1)=acf;
        pccf(epoch,1)=ccf;
      
   
end
figure(1),plot(1:epoch, pccf);
hold on;
% figure(2),plot(1:epoch,er);
% hold on;

%figure
%plot(1:epoch, pccf);
%hold on;
           
%end
 % classification of training data

a=0;
count=0;
for i=1:40
        for j=1:5
         a=a+1;
         %for input to hidden layer
            for h=1:nh
                sum=0;
                for k=1:100
                    sum=sum+(wih(k,h).*training1(i,j,k));
                end
                %Adding bias
                sum=sum+whb(1,h)*1;
                
                %Adding emotional neuron
                sum=sum+whm(1,h)*ypattrain1(i,j);
                
                yhnew(h,1)=1/(1+exp(-sum));
            end

            %For hidden to outer layer
              for z=1:op
                sum=0;
                for h=1:nh
                    sum=sum+(whj(h,z).*yhnew(h,1));
                end
                
                %Adding bias
                sum=sum+wjb(1,z)*1;
                
                %Adding emotional neuron
                sum=sum+whm(1,z)*ypattrain1(i,j);
                
                yjnew(1,z)=1/(1+exp(-sum));
              end
               answer(a,1)=find(yjnew==max(yjnew));
               if(label1train(a,answer(a,1))==1)
                   count=count+1;
               end
        end
               

end

% classification of testing data

a=0;
count1=0;
for i=1:40
        for j=1:5
         a=a+1;
%          for input to hidden layer
            for h=1:nh
                sum=0;
                for k=1:100
                    sum=sum+(wih(k,h).*testing1(i,j,k));
                end
%                 Adding bias
                sum=sum+whb(1,h)*1;
                
%                 Adding emotional neuron
                sum=sum+whm(1,h)*ypattest1(i,j);
                
                yhnew(h,1)=1/(1+exp(-sum));
            end

%             For hidden to outer layer
              for z=1:op
                sum=0;
                for h=1:nh
                    sum=sum+(whj(h,z).*yhnew(h,1));
                end
                
%                 Adding bias
                sum=sum+wjb(1,z)*1;
                
%                 Adding emotional neuron
                sum=sum+whm(1,z)*ypattest1(i,j);
                
                yjnew(1,z)=1/(1+exp(-sum));
              end
               answer(a,1)=find(yjnew==max(yjnew));
               if(label1test(a,answer(a,1))==1)
                   count1=count1+1;
               end
        end
               

end




