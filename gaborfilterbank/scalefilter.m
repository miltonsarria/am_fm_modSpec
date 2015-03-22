function b = scalefilter(b,f0)
%SCALEFILTER   Scale fitler to have passband approx. equal to one.
L=length(b);

b = b / abs( exp(-1i*2*pi*(0:L-1)*(f0/2))*(b.') );
