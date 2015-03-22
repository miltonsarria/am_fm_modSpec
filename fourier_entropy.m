function [Fec,T]=fourier_entropy(s,fs)
%s input signal
%fs sampling rate
%clear all
%[s,fs]=wavread('frf01_s01_solo.wav');
%s= resample(s, 16e3, fs);  fs=16e3;
%s=filter([1 -0.97], 1, s);
%s=asl_adjust(s,fs,-26); 
if fs~=16e3
    s =resample(s, 16e3, fs); %resample      
    fs=16e3;
end
s=filter([1 -0.97], 1, s); 
s=asl_adjust(s,fs,-26); 

tic
nfft=1024; %puntos nfft, twice the number of sample points
lw=32;
n=floor(fs*lw/1000); inc=round(n/2);
sf=enframe(s,hamming(n),inc,0); %enframe the input signal 
S=rfft(sf,nfft,2);
mc=melcepst(s,fs,'pdD',19,24,n,inc,0,0.5);
%[nframes,nsamp] = size(sf);
% for i=1:nframes
%   [rs,eta1] = xcorr(sf(i,:),'biased');
%   X(i,:)=rfft(rs,nfft); %power spectrum by definition, FT of autocorrelation function
% end

X=S.*conj(S); %|S|^2
%surf(log(p)','edgecolor','none')

%1-D ER
b1=[round((nfft/2+1)*450/(fs/2)):round((nfft/2+1)*650/(fs/2))];
b2=[round((nfft/2+1)*2800/(fs/2)):round((nfft/2+1)*3000/(fs/2))];

p1=X(:,b1)./repmat(sum(X(:,b1),2),1,size(X(:,b1),2)); %densidad de probabilidad en una banda especifica
h1=p1.*log(p1);
p2=X(:,b2)./repmat(sum(X(:,b2),2),1,size(X(:,b2),2)); %densidad de probabilidad
h2=p2.*log(p2);

H1=-sum(h1,2); %low
H2=-sum(h2,2); %high
ER=H2./H1;
%2-D SIE
b1=[round((nfft/2+1)*300/(fs/2)): round((nfft/2+1)*4150/(fs/2))];
b2=[round((nfft/2+1)*4150/(fs/2)): round((nfft/2+1)*8000/(fs/2))];
p1=X(:,b1)./repmat(sum(X(:,b1),2),1,size(X(:,b1),2)); %densidad de probabilidad
h1=p1.*log(p1);
p2=X(:,b2)./repmat(sum(X(:,b2),2),1,size(X(:,b2),2)); %densidad de probabilidad
h2=p2.*log(p2);
SIE=[-sum(h1,2),-sum(h2,2)];

%tilt
nc=20;
W  = 1 + (nc/2)*sin(pi/nc*(1:nc)');
p=20;
[nframes,nsamp] = size(sf);
id=1:2:nc;
m1=zeros(nframes,1);
c=zeros(nframes,nc);
for t=1:nframes
    ct  = zeros(nc,1);
    a  = zeros(nc,1);
    [rs1,eta1] = xcorr(sf(t,:)',p,'biased');
    % Calculo de los coeficientes LP basado en la recursion de durbin
    [a(1:p),xi1,kappa1] = durbin(rs1(p+1:2*p+1),p);
    %cepstral coeficients
    ct(1) = a(1);
    for (i = 2:nc)
        ct(i) = a(i) + (1:i-1)*(ct(1:i-1).*a(i-1:-1:1))/i;
    end
    c(t,:) = (ct.*W);
    m1(t)=-48/pi^3*sum(c(t,id)./(id.^2));
    %P=polyfit(0:nfft/2,10*log(X(t,:)),1);
    %m2(t)=P(1);
end

Enf=[ER,SIE,m1]; 
Fec=[mc, Enf];

T=toc;