function [len, vec] = harmonics(basefreq,t,mag,amp,samplerate)
% basefreq基频
% t持续时间
% mag整体幅度(0-1)
% amp向量，各谐波分量幅度(0-1)
% samplerate采样率
    len = floor(samplerate*t); % 本段音频应该有多少个采样点
    tvec = linspace(0,t,len); % 时间向量
    freqvec = 2*pi*basefreq*[1:length(amp)]'; % 频率向量 2pi*f_base*rate(倍率)
    mtx = freqvec*tvec; % 频率列向量*时间行向量=相位矩阵
    mtx = imag(exp(1i*mtx)); % 依据相位计算sin值
    vec = (mag*amp)*mtx; % 乘上各谐波幅度得最终音频
end