%example gabor
clear all
clc
%addpath(genpath('.........../voicebox'));
FS=8e3;
nFilters=23;
nTaps=512;
e_rms_bw=160;
lowF=10;
hiF=3800;
[fBankG cFr]= GaborFilterBank_erb(nFilters, nTaps, e_rms_bw, FS, lowF, hiF);
resp = 20*log10(abs(fft(fBankG')));

freqScale = (0:511)/512*FS;
semilogx(freqScale(1:256),resp(1:256,:));
%axis([0 FS/2 -50 0])
xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');


[fBankG cFr]= GaborFilterBank_mel(nFilters, nTaps, e_rms_bw, FS, lowF, hiF);

figure
freqScale = (0:511)/512*FS;
semilogx(freqScale(1:256),resp(1:256,:));
%axis([0 FS/2 -100 0])
xlabel('Frequency (Hz)'); ylabel('Filter Response (dB)');
