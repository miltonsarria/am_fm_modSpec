function gf = Gabor (alpha, ni, N)
	% gf = Gabor (alpha, ni, N)
	%
	%
	% pykfec - pyknogram frequency estimated coefficients toolbox for Matlab/Octave
	% Copyright (C) <2008>  <Marco.Grimaldi@gmail.com>
	%
	%
	% Creates a Gabor filter
	% Input:
	% 	alpha:		normalized effective RMS bandwidth
	%	ni:			(normalized) center frequency
	%	N:			number of filter's taps 
	% Output:
	%	gf:		vector 1 x N containing the discretized impulse response of the filter
	
	alpha = alpha*sqrt(2*pi);
	g = exp(-1*(alpha*[-N/2+1:N/2]).^2);
	c = cos(2*pi*ni*[0:N-1]);
	gf = g.*c/sum(g.*c);
end 