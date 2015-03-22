function [WIF]=am_fm_features(s,fs)
%This function computes the weighted instantaneous frequencies using a
%gammatone filter bank and the hilbert envelope approach for demodulation.
%The filter bank is already computed and saved in the file filt_bank.mat 
%Inputs:    
%       s    speech signal
%       fs   sampling rate, should be 16kHz,  if fs~=16e3 then the signal
%       is resampled to 16k
%Output:
%       WIF  the weighted instantaneous frequencies matrix, each row is an
%       observation. By default, the frames are 25 ms long and 10 ms overlap
%More information about WIF can be found in:
%M. Grimaldi and F. Cummins, “Speaker identification using instantaneous 
%frequencies,” IEEE Audio, Speech, Language Process., vol. 16, no. 6, 
%pp. 1097–1111, August 2008.
%
%Milton Sarria-Paja

%having voicebox toolbox, the filter bank can also be computed this way:
%nFilters=27;  lowF=100; hiF=7000;
%[B,A,~,~,gd]=gammabank(nFilters,FS,'mM',[lowF hiF]);
%filB.A=A; filB.B=B; filB.nfilt=nFilters; filB.gd=gd;
load('filt_bank.mat');

FS=16e3; 
if FS~=fs
    s       =resample(s, FS, fs); %resample      
end
%VAD
[~, ~, params]   = VQVAD; % Get default parameters to "params" struct
params.frame_len = 0.025;   % Frame duration, seconds
[speechInd]      = VQVAD(s, FS, params); VAD=speechInd==1;
if any(VAD)
        WIF              =compute_wif(s,FS,filB);
    	WIF=WIF(VAD,:);   
end
    
function [WIF]=compute_wif(y,FS,filB)

A       =filB.A;
B       =filB.B;
%gd      =filB.gd;

[Sc]=filterbank(B,A,y);          %apply the filterbank
ScH=hilbert(Sc)';                %analytic signal
[Mh,~,Fh] = moddecomphilb(ScH);  %demodulation
Fh=Fh*FS/2;
%lw=25; rw = 10; %window length, window rate in ms
n=0.025*FS; inc=0.01*FS;
%compute WIF
Fh=Fh/1e3;
Ai=Mh.*conj(Mh);
Fi=[];
for i=1:size(Mh,1)
    fai=enframe(Ai(i,:),n,inc); %enframe the envelope of the i-th chanel
    fF=enframe(Fh(i,:),n,inc);  %enframe the IF of the i-th chanel
    Num=diag(fai*fF');
    Den=sum(fai,2);
    Fi(i,:)=Num./Den;           %features of the i-th chanel 
end

WIF=-Fi';
clear Fh Fi Mh Ai fai fF Sc ScH Mh Fh