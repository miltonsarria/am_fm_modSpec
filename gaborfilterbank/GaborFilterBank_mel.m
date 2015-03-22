function [filterbank, fx] = GaborFilterBank_mel (nFilters, nTaps, e_rms_bw, SampleRate, lowF, hiF)
	% [filterbank centerFrequency]= GaborFilterBank (nFilters, nTaps, e_rms_bw, SampleRate), lowF, hiF)
	%
	%
	% pykfec - pyknogram frequency estimated coefficients toolbox for Matlab/Octave
	% Copyright (C) <2008>  <Marco.Grimaldi@gmail.com>
	%
	%
	% Creates a bank of Gabor filters linearly spaced beween lowF and HiF
	% Input and Outputs are as follows:
	% Input:
	% 	nFilters:		number of desired filters in the bank
	% 	nTaps:			number of taps in each filter
	% 	e_rms_bw:		Effective RMS BandWidth 
	%	SampleRate:		sample-rate of the signal to be processed
	%	lowF:			lowest frequency in the filterbank	
	%	hiF:			highest frequency in the filterbank
	%
	% Output:
	%	filterbank: 		matrix nFilters x nTaps containing the filterbank
	%	centerFrequency:	vector containing the (normalized) center frequency of each filter
	
    
    fx=[lowF;hiF];
    fx=frq2mel(fx); %convert limits to mel scale
    fx=linspace(fx(1),fx(2),nFilters)';     % centre frequencies in spacing units
    
    e_rms_bw_mel = frq2mel(1000 + e_rms_bw/2) - frq2mel(1000 - e_rms_bw/2);
    	
    e_rms_bw_hz = mel2frq( fx + e_rms_bw_mel/2 ) - mel2frq(fx-e_rms_bw_mel/2);
		
	alpha = e_rms_bw_hz./SampleRate;
    
    fx=mel2frq(fx)/SampleRate; % centre frequencies back to  Hz
    
    filterbank = zeros (nFilters,nTaps);
	for i = 1:nFilters
		filterbank(i,:) = Gabor(alpha(i),fx(i),nTaps);
        filterbank(i,:) = scalefilter(filterbank(i,:) ,fx(i)/0.5);
	end
end