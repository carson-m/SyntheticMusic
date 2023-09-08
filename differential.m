function vec = differential(dat,n)
%功能与自带的diff一样，只是在最后补终值保证vec长度与dat长度一致
    vec = diff([dat;dat(end-n+1:end,1)],n);
end