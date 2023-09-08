function [num, mag, basefreq, harmo] = toneanalyse(ndat,nFs,freqs,maxhmnc)
%num基波序号 mag音符幅度 basefreq基频 harmo各次谐波系数向量 dat待分析数据 freqs涉及到的各种频率 Fs采样率
%maxhmnc 取的最高次谐波
    %subplot(2,1,1);
    %plot(ndat);
    % dat = resampledenoise(dat);
    % subplot(3,1,2);
    % plot(dat);
    % ndat = repmat(dat,[100,1]);
    %nFs = Fs; %采样率
    mag = max(ndat);
    nL = length(ndat); %信号长度
    nF = fft(ndat);
    nFP2 = abs(nF/nL);
    nFP1 = nFP2(1:(floor(nL/2)+1));
    nFP1(2:end-1) = 2*nFP1(2:end-1);
    nf = nFs*(0:floor(nL/2))/nL; %频率轴
    %subplot(2,1,2);
    %plot(nf,nFP1);
    [ampmax,idxmax] = max(nFP1);
    vec2proc = nFP1-0.3*ampmax; %提取出来的峰值至少为0.3倍的最大幅值，确保基频选择较为准确
    vec2proc(vec2proc<0)=0; %将幅频曲线中幅度较小的剔除，避免造成对峰值提取与分析的干扰
    [~,locs] = findpeaks(vec2proc);
    locstmp = locs(locs<idxmax);
    if isempty(locstmp)
        basefreq = nf(idxmax);
        harmo = 1;
    else
        guesstmp=nf(idxmax)./nf(locstmp);
        guesstmp=abs(guesstmp-round(guesstmp));
        %[dif,candidate]=min(guesstmp);
        candidates=locstmp(guesstmp<0.02);
        % if dif>0.01
        %     basefreq = nf(idxmax);
        %     harmo = 1;
        % else
        %     basefreq = nf(locs(candidate));
        %     if abs(basefreq/nf(idxmax)-1)<0.015
        %         basefreq = nf(idxmax);
        %         harmo = 1;
        %     else
        %         harmo = nFP1(locs(candidate))/ampmax;
        %     end
        % end
        if isempty(candidates)
            basefreq = nf(idxmax);
            harmo = 1;
        else
            basefreq = nf(candidates(1));
            if abs(basefreq/nf(idxmax)-1)<0.015 %如果原始片段点数较少，则频域分辨率较差，可能把最大值附近的点当作新的基频，故额外加以判断
                basefreq = nf(idxmax); %若新的基频和最大幅度对应的频率差别过小，则直接将后者作为基频
                harmo = 1;
            else
                harmo = nFP1(candidates(1))/ampmax;
            end
        end
    end
    % [autocor,~] = xcorr(nFP1,'coeff');
    % [~,lclg] = findpeaks(autocor,'MinPeakheight',0.3*max(autocor));
    % plot(autocor);
    % axis([0 1000 -1 1])
    % f=mean(diff(lclg))*nFs/nL;
    [~,loc] = min(abs(freqs-basefreq));
    basefreq = freqs(loc);
    num = loc;
    for x=2:maxhmnc
        thresh = floor(0.015*nL*x*basefreq/nFs); %允许在误差不超过+-1.5%的范围内搜索峰值
        freqtmp = x*basefreq;
        if freqtmp>=nFs/2
            harmo=[harmo,zeros(1,maxhmnc-x+1)];
            break
        end
        [~,locharm] = min(abs(nf-freqtmp)); %寻找最接近谐波频率的现有频率的位置locharm i.e.location of harmonic
        ampharm = max(nFP1(max(locharm-thresh,1):min(locharm+thresh,end))); %该谐波分量的幅度
        harmo = [harmo,ampharm/ampmax];
    end
end

