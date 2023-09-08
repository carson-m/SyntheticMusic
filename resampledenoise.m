function sample = resampledenoise(dat)
    lendat = length(dat);
    upsample = resample(dat,10,1);
    meansample = upsample(1:lendat);
    for x = 1:9
        meansample = meansample + upsample(x*lendat+1:(x+1)*lendat);
    end
    meansample = meansample/10;
    sample = repmat(meansample,[10,1]);
    sample = resample(sample,1,10);
end