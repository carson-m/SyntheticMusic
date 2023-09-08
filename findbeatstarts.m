function vec=findbeatstarts(music)
% music:对其分析各音符开始位置
    [~,locs]=findpeaks(music);
    vec=[];
    lastvalloc=-300;
    for x=1:length(locs)
        if(locs(x)-lastvalloc<300)
            continue
        else
            vec=[vec,locs(x)];
            lastvalloc=locs(x);
        end
    end
end