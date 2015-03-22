function [feats,filB]=all_feats(s,filB,FS,FR)
%s input signal
%filB filter bank structure
%FS sampling rate
%FR frquency rangue in Hz for the filter bank
%list_wav
%fin
n=0.025*FS; hop=0.01*FS;
lowF=FR(1); 
if FR(2)==8000, hiF=7000;
else hiF=FR(2);
end
Gab=filB.Gabor;
Gam=~filB.Gabor;
nFilters=filB.nfilt;
nTaps=filB.nTaps;
e_rms_bw=filB.e_rms_bw;
scale=filB.scale;
if strcmp(scale,'m') 
    [B,A,~,~,gd]=gammabank(nFilters,FS,'mM',[lowF hiF]);
    [fBankG, ~]= GaborFilterBank_mel(nFilters, nTaps, e_rms_bw, FS, lowF, hiF);
else  
     [B,A,~,~,gd]=gammabank(nFilters,FS,'',[lowF hiF]);
     [fBankG,~]= GaborFilterBank_erb(nFilters, nTaps, e_rms_bw, FS, lowF, hiF);
end
 
filB.Gabor= fBankG;
filB.A=A;
filB.B=B;
filB.gd=gd;


[~, ~, params]   = VQVAD; % Get default parameters to "params" struct
params.frame_len = 0.025;   % Frame duration, seconds
[speechInd]      = VQVAD(s, FS, params); VAD=speechInd==1;
%% WIF
[WIF,MHEC]=compute_amfm(s,FS,filB,n,hop,Gab,Gam);      
feats.WIF=WIF;
feats.MHEC=MHEC;


%%
function [WIF,MHEC]=compute_amfm(y,FS,filB,n,inc,Gab,Gam)

A       =filB.A;
B       =filB.B;
%gd      =filB.gd;
fBankG=filB.Gabor;
nFilters=filB.nfilt;
if Gam
    [Sc]=filterbank(B,A,y);          %apply the filterbank
    ScH=hilbert(Sc)';                %analytic signal
elseif Gab
    Sc=zeros(nFilters,length(y));
    for f = 1:nFilters %apply the filterbank
        Sc(f,:)= filter(fBankG(f,:), 1, y);
    end
    ScH=hilbert(Sc')';  
end

[Mh,~,Fh] = moddecomphilb( ScH );   %demodulation
Fh=Fh*FS/2;
%WIF
Fh=Fh/1e3;
Ai=Mh.*conj(Mh);
Fi=[];
for i=1:size(Mh,1)
    fai=enframe(Ai(i,:),n,inc);
    fF=enframe(Fh(i,:),n,inc);
    Num=diag(fai*fF');
    Den=sum(fai,2);
    Fi(i,:)=Num./Den;    
end

WIF=-Fi';

%MHEC
nc=size(ScH,1); %number of channels
%hilbert envelope
Ev=abs(ScH);
E=[];
for i=1:nc %across each channel    
    Eci=enframe(Ev(i,:),hamming(n),inc); %enframe the bandpass signal
    Elog=log(mean(Eci,2));
    E=[E; Elog'];
end
E=E./repmat(mean(E),size(E,1),1);
MHEC=rdct(E).';
MHEC(:,1)=[];

