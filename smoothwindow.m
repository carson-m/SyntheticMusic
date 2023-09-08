function vec = smoothwindow(music,windowsize,offset)
    %music待处理音乐波形 windowsize为Hann平滑窗大小，取奇数
    %offset提供偏置，可移动平滑窗中心位置以应对不对称的包络，正数为右移，取0时平滑窗居中，最多移floor(windowsize/2)
    len = length(music); %要平滑的音频的长度
    hwindsize = floor(windowsize/2); %half windowsize;
    music = [zeros(windowsize-1,1);music;zeros(windowsize-1,1)]; %开头与末尾补零
    %wind = sum(music(windowsize+1-hwindsize+offset:windowsize+1+hwindsize+offset,1)); %初始化
    Hwind = hann(windowsize); %生成Hanning窗
    vec = zeros(len,1);
    %vec(1,1) = wind;
    for x=1:len
        %wind = wind-music(windowsize+x-1-hwindsize+offset,1)+music(windowsize+x-1+hwindsize+offset,1);
        vec(x,1) = music((windowsize+x-hwindsize+offset):(windowsize+x+hwindsize+offset),1)'*Hwind;
    end
end