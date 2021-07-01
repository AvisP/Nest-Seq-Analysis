load('I:\avgn_paper-vizmerge\Nest3_from_python.mat');

for i =1:size(indv,1)
    labels(i) = {char(strcat(convertCharsToStrings(key(i,:)),'_',num2str(i)))};
    keywords(i) =  {convertCharsToStrings(indv(i,:))};
end

timeSeriesData = audio;

save('Nest3.mat', 'keywords', 'labels', 'timeSeriesData')

TS_Init('Nest3.mat','INP_mops.txt','INP_ops.txt',0,'HCTSA_Nest3.mat');
sample_runscript_matlab(1,10,'HCTSA_Nest3.mat');

TS_Init('temp.mat','INP_mops.txt','INP_ops.txt',0,'HCTSA_temp.mat');
sample_runscript_matlab(1,1,'HCTSA_temp.mat');