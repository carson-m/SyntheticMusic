function [len, vec] = env(t,samplerate)
    % 包络生成器
    % t:持续时长
    % samplerate: 采样率
    len = floor(samplerate*t); %根据时间与采样率获得包络长度
    %vec = 0.7*linspace(0,30,len)./exp(linspace(0,10,len));
    vec = 0.7*((linspace(0,1,len)).^0.33)./exp(linspace(0,10,len)); % 包络函数
end