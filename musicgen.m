function vec = musicgen(freqs,musicdat,samplerate,xfadetime,lenequal)
%lenequal各音符是否等长 输入正值使每个音符都持续该特定时长(s)，非正值则各音之间只有短暂的crossfade，建议lenequal<2
%lenequal为正时xfadetime失效
totaltime = sum(musicdat(:,1));
vec = zeros(1,3*samplerate+ceil(totaltime*samplerate)); %每首歌开始前一秒安静，后面留出两秒空余
pamt = floor(xfadetime*samplerate); %amount of points 渐变涉及多少个采样点
currIndex = samplerate; %记录当前生成了多少个有效的点
limitMag = 0.5; %防止总幅度过1导致失真，乘一系数
if lenequal <= 0
    for x = 1:1:length(musicdat(:,1)) %此循环用于自动加入迭接，使音符之间连贯
        timetemp = musicdat(x,1)+xfadetime; %音乐矩阵的第一列为音符持续时长
        if musicdat(x,2) ~= 0 %音乐矩阵的第二列为音符序号
            [notelen,notetmp] = harmonics(freqs(musicdat(x,2)),timetemp,limitMag*musicdat(x,3),musicdat(x,4:end),samplerate);
            [~,envelope] = env(timetemp,samplerate);
            notetmp = notetmp.*envelope;
        else
            [notelen,notetmp] = gapgen(timetemp,samplerate); %如果音符序号为0，代表一个空拍
        end
        vec(currIndex-pamt+1:currIndex-pamt+notelen)=vec(currIndex-pamt+1:currIndex-pamt+notelen)+notetmp; %将当前音符接入已有音乐末尾
        currIndex=max(currIndex,currIndex-pamt+notelen); %此判断在当前不必须，为日后扩展留余地，即有可能某个音符的时间段完全被另一个音符的时间段所包含
    end
else
    for x = 1:1:length(musicdat(:,1)) %此循环用于自动加入迭接，使音符之间连贯
        if musicdat(x,2) ~= 0 %音乐矩阵的第二列为音符序号
            [notelen,notetmp] = harmonics(freqs(musicdat(x,2)),lenequal,limitMag*musicdat(x,3),musicdat(x,4:end),samplerate);
            [~,envelope] = env(lenequal,samplerate);
            notetmp = notetmp.*envelope;
        else
            [notelen,notetmp] = gapgen(lenequal,samplerate); %如果音符序号为0，代表一个空拍
        end
        legallen = floor(musicdat(x,1)*samplerate); %有效长度
        vec(currIndex+1:currIndex+notelen)=vec(currIndex+1:currIndex+notelen)+notetmp; %将当前音符接入已有有效音乐末尾
        currIndex = max(currIndex,currIndex+legallen);
    end
end
end