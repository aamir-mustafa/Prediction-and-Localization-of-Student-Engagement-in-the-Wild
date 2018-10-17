first= values(33,:);
f=first*3;
%f(148:150)=0.8;
 %f(1:3)=0.8;
f= smooth(f, 15);

plot(f,'LineWidth',1.5);
ylim([0 3])
