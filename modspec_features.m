function [Y, E, Ei,ym]=modspec_features(s,fs,B,M,fl,fh,nf)
%Milton Orlando Sarria Paja
%INRS-EMT
%This function computes the features designed for whispered speech
%detection in noisy environments, this was evaluated with bable noise
%Inputs:
%       s   input signal
%       fs  sampling rate, ( 16khz)
%       B   Hz, this is the bandwidth to determine what is the max
%       modulation frequency we want to detect (63 hz)
%       M   sec. this is the length of the windows in the freq domain (0.1
%       secs)
%       fl  lower frequency  to limit the range of freq, (0) 
%       fh  higher frequency (0.5 (fs/2) )
%Outputs:
%       Y   features for whispered speech detection
%       E   mod spectrum per fft bin or filter in the acoustic domain
%       Ei  mod spectrum after filtering in the modulation domain
%       ym log energy per band -- output of filter banks in te auditory frequency
%B=63; M=0.1; fl=0; fh=0.5; nf=24; out=0;  s=filter([1 -0.97], 1, s);  s=asl_adjust(s,fs,-26);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Whispered Speech Detection in Noise Using Auditory-Inspired Modulation Spectrum Features
%Sarria-Paja, M.; Falk, T.H., Signal Processing Letters, IEEE , vol.20, 
%no.8, pp.783,786, Aug. 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if fs~=16e3
    s =resample(s, 16e3, fs); %resample      
    fs=16e3;
end
%pre-emphasis and power normalization
s=filter([1 -0.97], 1, s); 
s=asl_adjust(s,fs,-26); 
        
if ~exist('B','var'),   B=63; end
if ~exist('M','var'),   M=0.1; end
if ~exist('fl','var'),  fl=0; end
if ~exist('fh','var'),  fh=0.5; end
if ~exist('nf','var'),  nf=24; end


L=ceil(2*fs/B);                 %window size for the given B
nfft=1024;                      %number of fft bins (nfft/2)
fr=2*B;                         %frame rate, frames per second
inc1=ceil((fs-L)/(fr-1));       %calculate the increment for the next window
sf=enframe(s,hamming(L),inc1);  %enframe the signal
S=rfft(sf,nfft,2);              %real fft  
S=S';                                       %nfft/2+1 x number of frames, each column is a frame, each row a fft bin
[m,a,b,c]=melbankm(nf,nfft,fs,fl,fh,'u');   %
fa=a; a=b; b=c;                           
pw=S(a:b,:).*conj(S(a:b,:));                %power domain
pth=max(pw(:))*1E-20;  
ym=max(m*abs(S(a:b,:)),sqrt(pth));  %
ym=log(ym);
mS= ym';                                   %number of frames x number of mel filters  

%process in the modulation domain
n=fix(M*fr); inc2=ceil(1/3*n);  %compute window size and overlap
nfft2=256;                      %128 bins in the modulation domain
%to know the size of the output then the enframe is done for the first bin
sff=enframe(mS(:,1),hamming(n),inc2);           %
E=zeros(size(mS,2), nfft2/2+1,size(sff,1));     %cambiar los fft bins en caso de que cambien a los filtros
%calculate mod spectrum using nfft2 bins
for i=1:size(mS,2) %across columns - dft bins or filters
    sff=enframe(mS(:,i),hamming(n),inc2,0); %# frames x n points (M secs)
    Sf=rfft(sff,nfft2,2);                   %# frames (i-th bin) x nfft2/2 points (fft bins in mod frequency)   
    Sf=abs(Sf);            %magnitude spectrum of i-th fft bin, |Sf|, each row a frame
     for j=1:size(Sf,1)
         E(i,:,j)=Sf(j,:);
     end
end

% features
Y=[];Ei=[];% 
fl=1/fr; fh=0.5;
nf=8;
[m,a,b,c]=melbankm(nf,nfft2,fr,fl,fh,'ul'); %filter bank 2 for mod spec
fm=a; a=b; b=c; %fc=find(10.^fc<20);
fa=mel2frq(fa); fai=find(fa>1e3);

for i=1:size(E,3)
    me=E(:,:,i)'; %fft bins x number of filters in acoustic domain
    me=me(a:b,:); 
    pth=max(me(:))*1E-20;
    ym=max(m*me,sqrt(pth));
    ym=ym(10.^fm<24,:);
    Ei(:,:,i)=ym';
    ym=log(ym);
    tilt=[];
    for m=1:size(ym,1)
         P=polyfit(fa(fai),ym(m,fai),1);
         tilt(m)=P(1);
    end  
    Y(:,i)=[ym(:);tilt(:)];%C(:); 
end

Y=Y';

