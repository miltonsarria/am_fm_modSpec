function [speechInd, LLR, params] = VQVAD(s, fs, params);

% Adaptive voice activity detector (VAD) presented in,
% 
% [1]   Tomi Kinnunen and Padmanabhan Rajan, "A practical, self-adaptive voice 
%       activity detector for speaker verification with noisy telephone and
%       microphone data", to appear in ICASSP 2013, Vancouver.
%
% Inputs: 
% 
%   s           Column vector of PCM signal samples as obtained through "wavread"
%               function of MATLAB.
%   fs          Sampling rate in Hertz
%   params      Optional struct of magic control parameters. To run the VAD
%               using the default parameters, call with the first two arguments
%               only:
%
%                   [speechInd, LLR] = VQVAD(s, fs);
%
%               To see what the control parameters are, call without any arguments: 
%           
%                   [junk, junk, params] = VQVAD;  
%
%               You may also provide your custom parameters,
%       
%                   [speechInd, LLR] = VQVAD(s, fs, params);
%
%               Especially the minimum energy threshold (params.min_energy) may
%               require some fine-tuning depending on whether you work on telephone
%               or microphone data. According to our experience with NIST
%               speaker recognition data, values in the range [-75 dB ..-55 dB]
%               seem to give reasonable results on "typical" NIST SRE data. 
%
% Outputs:      
%
%   speechInd   The binary VAD labels
%   LLR         Log-likelihood ratios (i.e. just nonspeech - speech
%               distance) in the case you wanna use different thresholding.
%               SpeechInd is just LLR threshold at zero.
%   params      Struct of parameter values
%
% NOTE!         This code should be otherwise self-contained, but it does
%               use the spectral subtraction routines from Voicebox 
%               (http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html)
%               So download Voicebox and add to your Matlab path.
%
% (C) Copyright School of Computing (SoC), University of Eastern Finland (UEF)
% Use this code completely at your own risk. 
%
% Feedback is always welcomed!
% E-mail: tkinnu@cs.uef.fi and paddy@cs.uef.fi.

% A couple of control parameters with default values
if (nargin < 3)
    params.frame_len        = 0.030;   % Frame duration, seconds
    params.frame_shift      = 0.0100;   % Frame hop, seconds
    params.dither           = true;     % Add neglible small-amplitude Gaussian noise to avoid duplicated MFCC vectors in zero-only parts
    params.clean_energy     = true;     % Apply spectral subtraction on energy VAD values
    params.clean_MFCCs      = false;    % Apply spectral subtraction also to MFCCs?
    params.energy_fraction  = 0.10;     % Fraction of high/low energy frames picked for speech/nonspeech codebook training
    params.min_energy       = -75;      % Minimum-energy constraint 
    params.vq_size          = 16;       % Codebook size (# codewords) for speech and nonspeech
    params.max_kmeans_iter  = 20;       % Maximum number of k-means iterations
    params.num_filters      = 27;       % Number of mel bands
    params.num_cep          = 12;       % Number of MFCCs
    params.include_C0       = true;     % Include C0 coefficient (seems like a good idea for VAD purposes)
    params.NFFT             = 512;      % FFT size
end;

% Just return the default parameters
if (nargin == 0)
    speechInd = [];
    LLR = [];
    return;
end;

% Check that we don't feed in stereo data
if size(s, 2) > 1
    error(sprintf('Expecting single-channel audio'));
end;

% Compute value from s that we will use to initialize random number
% generation, to make sure we get the SAME result if we repeat this twice
% for the same signal (randomization is used both in "dithering" and K-means
% implementation.
seed = floor(max(abs(s)) * length(s) + abs(min(s))) + 1;
rng(seed);

% Convert framing parameters from seconds into samples, for this
% samplerate.
frame_len_samp   = round(params.frame_len   * fs);
frame_shift_samp = round(params.frame_shift * fs);

% "Dithering" to avoid having identical training vectors in our speech and nonspeech
% codebooks (yes, it can actually be a problem)
if params.dither
    s = s + (1e-09).*randn(length(s), 1);
end;

% Optional speech cleaning to enhance energy VAD initialization
if params.clean_energy
    % Define Wiener filter parameters
    pp.g      = 2;
    pp.e      = 2;
    pp.ne     = 1;
    pp.am     = 10; % allow aggressive oversubtraction
    s_cleaned = specsub(s, fs, pp);
    
    % Extract frame energy values from cleaned frames
    frames = enframe(s_cleaned, boxcar(frame_len_samp), frame_shift_samp);
    energy = 20*log10(std(frames')+eps);
else
    % Extract frame energy values from original (noisy) frames
    frames = enframe(s, boxcar(frame_len_samp), frame_shift_samp);
    energy = energy_orig; % This was already computed
end;

% Extract MFCCs either from cleaned or original (noisy) signal
if params.clean_MFCCs
    [Cep, frames] = compute_MFCCs(s_cleaned, fs, params);
else
    [Cep, frames] = compute_MFCCs(s, fs, params);
end;
nf = size(Cep, 1);

% Rank all frame energies and determine the lowest and highest energy
% frames.
[junk, frame_idx] = sort(energy);
nonspeech_frames  = frame_idx(1:round(nf * params.energy_fraction));
speech_frames     = frame_idx(nf - round(nf * params.energy_fraction):end);

% Train the speech and nonspeech models from the MFCC vectors corresponding
% to the highest and lowest frame energies, respectively
speech_model    = train_codebook(Cep(speech_frames, :), params);
nonspeech_model = train_codebook(Cep(nonspeech_frames, :), params);

% Compute VQ distances to speech and nonspeech
D_speech    = pdist2(Cep, speech_model, 'euclidean').^2;
D_nonspeech = pdist2(Cep, nonspeech_model, 'euclidean').^2;
LLR         = min(D_nonspeech') - min(D_speech');

% Threshold arbitrarily at LLR==0
VQ_speechInd = (LLR >= 0);

% Accept ONLY if the energies are also high enough
speechInd = VQ_speechInd; 
speechInd = VQ_speechInd & (energy >= params.min_energy);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Cep, frames] = compute_MFCCs(s, fs, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

frame_len       = round(params.frame_len * fs);
frame_shift     = round(params.frame_shift * fs);
num_filters     = params.num_filters;
num_cep         = params.num_cep;
NFFT            = params.NFFT;
Fmin_Hz         = 0;
Fmax_Hz         = fs/2;
include_C0      = params.include_C0;

% Extract base MFCCs
[FB, junk]      = Mel_Bank(num_filters, fs, Fmin_Hz, Fmax_Hz, NFFT, 0);
win_fun         = hamming(frame_len);
frames          = enframe(s, boxcar(frame_len), frame_shift);
windowed_frames = frames .* repmat(win_fun', size(frames,1),1);
Cep             = ComputeFFTCepstrum(windowed_frames, FB', num_cep, NFFT, include_C0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = train_codebook(X, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if size(X, 1) <= params.vq_size
    C = X; % Not enough training vectors, just return the original data
else
    [C,junk] = my_kmeans(X, params.vq_size, params.max_kmeans_iter);
end;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Cep = ComputeFFTCepstrum(Frames, FilterBank, NumCoeffs, NFFT, keep_C0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ESpec       = abs(fft(Frames',NFFT)).^2;
LogSpec     = log(ESpec(1:NFFT/2+1, :) + 1e-11);
Cep         = dct(LogSpec);
if keep_C0
    Cep = Cep(1:NumCoeffs, :)';
else
    Cep = Cep(2:NumCoeffs+1, :)';
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H, CenterFreq] = Mel_Bank(NumFilters, fs, FminHz, FmaxHz, NFFT, PlotResponse);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Some messy, more than decade-old code to create typical MFCC filterbank.
% No guarantees that it's the optimum implementation of MFCC bank... 
% Use completely at your own risk ;-)

NumFilters = NumFilters + 1;
for m=2:NumFilters+1
    CenterFreq(m-1) = Mel2Hz(Hz2Mel(FminHz) + m*((Hz2Mel(FmaxHz) - Hz2Mel(FminHz))/(NumFilters + 1)));
    f(m)            = floor((NFFT/fs) * CenterFreq(m-1));
end;
f(1) = floor((FminHz/fs)*NFFT)+1;
f(m+1) = ceil((FmaxHz/fs)*NFFT)-1;

H = zeros(NumFilters,NFFT/2+1);
for m=2:NumFilters
    for k=f(m-1):f(m)
        foo = f(m)-f(m-1);
        if (foo==0)
            foo = 1;
        end;
        H(m-1,k) = (k - f(m-1))/foo;
    end;
    for k = f(m):f(m+1)
        foo = f(m+1) - f(m);
        if (foo==0)
            foo = 1;
        end;
        if (f(m+1) - f(m) ~= 0)
            H(m-1,k) = (f(m+1) - k)/foo;
        end;
    end;
end;
H = H(1:NumFilters-1,:)';
CenterFreq = f(2:end);

if (PlotResponse)
    plot(((0:NFFT/2)./NFFT).*fs,H);
    xlabel('Frequency [Hz]');
    ylabel('Gain');
    grid on;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Mel = Hz2Mel(f_Hz);
Mel = (1000/log10(2)) * log10(1 + f_Hz/1000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f_Hz = Mel2Hz(f_mel);
f_Hz = 1000 * (10^((log10(2) * f_mel)/1000) - 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [C, E] = my_kmeans(X, K, max_iter);
% Good old k-means, a version that hopefully doesn't crash.

% Special case, just one centroid
if (K == 1)
    if (size(X, 1) > 1)
        C = mean(X);
    else
        C = X;
    end;
    dist = pdist2(X, C, 'euclidean').^2;
    E    = mean(dist);
    return;
end;

% Otherwise, first pick K code vectors at random as initial centroids
P = randperm(size(X, 1));
C = X(P(1:K), :);

% Continue iterating K-means
conv_threshold = 1e-5;
for it = 1:max_iter
    
    % First, compute squared euclidean distance to cluster centroids
    dist = pdist2(X, C, 'euclidean').^2;
    
    % Find index of nearest centroid for each data point
    [min_dist, min_idx] = min(dist');
    
    % Record the mean square error (MSE), used for convergence check
    E(it) = mean(min_dist);
    
    % ----- Centoid updates -----
    for k = 1:K
        idx = find(min_idx == k);
        if (length(idx) == 0) % Empty cluster, pick random data point from X as centroid
            C(k, :) = X(floor(rand * size(X, 1))+1, :);
        elseif (length(idx) == 1) % Just one vector, avoid forming scalar average
            C(k, :) = X(idx, :);
        else % Normal case, take mean vector
            C(k, :) = mean(X(idx, :));
        end;
    end;
    
    % Check for relative change in MSE
    if it > 1
        converged = (abs((E(it) - E(it-1))/E(it-1)) < conv_threshold);
        if converged
            break;
        end;
    end;
    
end;