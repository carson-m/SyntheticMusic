function [len,vec] = gapgen(t,samplerate) %gap generator 生成对应时长的空音
    len = floor(samplerate*t);
    vec = zeros(1,len);
end