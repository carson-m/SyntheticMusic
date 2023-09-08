function vec = appendnote(music,appnote,xfadetime,samplerate)
%music已有音乐 appnote串接音符 xfadetime渐变时长 samplerate采样率
    pamt = floor(xfadetime*samplerate); %amount of points 渐变涉及多少个采样点
    vec = [music(1:end-pamt),music(end-pamt+1:end)+appnote(1:pamt),appnote(pamt+1:end)];
end