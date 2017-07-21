#!/usr/bin/env python3

import numpy as np
from scipy import fftpack
from scipy import interpolate
import matplotlib.pyplot as plt

def linear_itp(data,t,interval=0.001):
    """
    对输入的时间序列进行线性插值.

    参数
    ----
    data : array_like
        待插值的序列
    t : array_like
        data 序列对应的时刻,单位为 s ,若 data 的长度和 t 的长度不相等将会报错
    interval : float, optional
        插值间隔,默认为 0.01

    返回
    ----
    插值后的 data 和 t 两个序列 data_new , t_new

    """
    if len(data)!=len(t):
        msg="data 和 t 长度不匹配!"
        raise Exception(msg)

    t_new=np.arange(t[0],t[-1],interval)
    t_new=np.append(t_new,t[-1])
    f_linear=interpolate.interp1d(t,data)
    data_new=f_linear(t_new)

    return(data_new,t_new)

def bspline_itp(data,t,interval=0.001):
    """
    对输入的时间序列进行样条插值.

    参数
    ----
    data : array_like
        待插值的序列
    t : array_like
        data 序列对应的时刻,单位为 s ,若 data 的长度和 t 的长度不相等将会报错
    interval : float, optional
        插值间隔,默认为 0.01

    返回
    ----
    插值后的 data 和 t 两个序列 data_new , t_new

    """
    if len(data)!=len(t):
        msg="data 和 t 长度不匹配!"
        raise Exception(msg)

    t_new=np.arange(t[0],t[-1]+interval,interval)
    tck=interpolate.splrep(t,data)
    data_new=interpolate.splev(t_new,tck)

    return(data_new,t_new)

def pre_treat(tr,dist,samp,vmin,vmax):
    print('************************************************************')
    #对折
    tr_left=list(tr[:int((len(tr)+1)/2)])
    tr_left.reverse()
    tr_right=tr[int((len(tr)-1)/2):]
    tr=(np.array(tr_left)+np.array(tr_right))/2
    tr_full=tr.copy()
    #按照 vmin 和 vmax 截取 tr
    index1=int(samp*dist/vmax)
    index2=int(samp*dist/vmin+0.5)
    tr=tr[index1:index2+1]
    t_tr=np.linspace(index1/samp,index2/samp,len(tr))
    print('最小到时:',t_tr[0],'最大到时:',t_tr[-1])
    #计算速度刻度
    vel_scale=dist/t_tr

    return(tr_full,tr,t_tr,vel_scale)

def compute_group_vel(ccf_full,ccf,t_ccf,dist,tmin,tmax,vmin,vmax,alpha):
    #构建中心周期序列
    #per=np.arange(tmin,tmax+0.1,0.1)
    delta=0.1
    per=[]
    cper=tmin
    while cper<tmax:
        per.append(cper)
        cper+=delta
        delta*=1.01
    if per[-1]<tmax: per.append(tmax)
    #将 ccf 变换到频率域
    ccf_freq=fftpack.ifft(ccf)
    freq_samp=2*np.pi*abs(fftpack.fftfreq(len(ccf)))
    ccf_full_freq=fftpack.ifft(ccf_full)
    freq_full_samp=2*np.pi*abs(fftpack.fftfreq(len(ccf_full)))
    #时频分析
    SSNR=[]
    t_ariv=[]
    ph_ariv=[]
    per_inst=[]
    omg_inst=[]
    group_vel=[]
    ftan_mat=[]

    for tn in per:
        omgn=2*np.pi/tn
        #窄带滤波
        ccf_freq_nbG=ccf_freq*np.exp(-alpha*((freq_samp-omgn)\
        /omgn)**2)
        ccf_full_freq_nbG=ccf_full_freq*\
        np.exp(-alpha*((freq_full_samp-omgn)/omgn)**2)
        #变换到时间域
        ccf_time_nbG=fftpack.fft(ccf_freq_nbG).real
        ccf_full_time_nbG=fftpack.fft(ccf_full_freq_nbG).real
        ccf_time_nbG_hilbert=fftpack.ihilbert(ccf_time_nbG)
        #计算包络
        env=np.sqrt(ccf_time_nbG**2+ccf_time_nbG_hilbert**2)
        #amp=20*np.log10(env)    #单位:DB
        amp=env.copy()
        [env,t_env]=bspline_itp(env,t_ccf)
        #计算相位
        phase=np.arctan(ccf_time_nbG_hilbert/ccf_time_nbG)
        for i in range(len(phase)-1):
            k=int(abs(phase[i+1]-phase[i])/np.pi+1/2)
            phase[i+1]=phase[i+1]+np.pi*k
        [phase,t_phase]=linear_itp(phase,t_ccf)
        #计算瞬时频率
        index=np.where(env==np.max(env))[0][0]
        try:
            omgn=(phase[index+1]-phase[index])/(t_phase[index+1]-t_phase[index])
        except:
            continue
        if omgn<0:
            continue
        omg_inst.append(omgn)
        t_ariv.append(t_env[index])
        while phase[index]>np.pi/2:
            phase[index]=phase[index]-np.pi
        ph_ariv.append(phase[index])
        #计算群速度
        per_inst.append(float(str(round(2*np.pi/omg_inst[-1],3))))
        group_vel.append(dist/t_ariv[-1])
        ftan_mat.append(list(amp))
        #计算谱信噪比
        index1=int(dist/vmax)
        index2=int(dist/vmin)
        signal=ccf_full_time_nbG[index1:index2+1]
        noise=ccf_full_time_nbG[index1+1000:index2+1001]
        noise_rms=np.sqrt(np.sum(noise**2)/len(noise))
        SSNR.append(np.max(signal)/noise_rms)
        
        #print(tn,2*np.pi/omgn)
        #plt.subplot(211)
        #plt.plot(t_ccf,ccf,'k')
        #plt.subplot(212)
        #plt.plot(amp,'k')
        #plt.show()
    ftan_mat=np.transpose(ftan_mat)
    ftan_mat=100*ftan_mat/np.max(ftan_mat)
    #调整振幅(开方乘十),为了画图好看
    tmp=[]
    for line in ftan_mat:
        tmp.append(np.sqrt(np.sqrt(line)*10)*10)
    ftan_mat=tmp.copy()
    
    #plt.plot(per_inst,SSNR)
    #plt.show()

    return(ftan_mat,group_vel,per_inst,omg_inst,t_ariv,ph_ariv,SSNR)

def compute_phase_vel(group_vel,per_inst,omg_inst,dist,t_ariv,ph_ariv,PHV):
    #查找最大周期对应的相速度预测值
    pper=[]
    pphv=[]
    for pp in PHV:
        pper.append(pp[0])
        pphv.append(pp[1])
    [pphv,pper]=bspline_itp(pphv,pper)
    for i in range(len(pper)):
        if str(pper[i])==str(per_inst[-1]):
            index=i
    #print(pper[index],per_inst[-1])
    #根据预测值计算真实相速度
    phase_vel=[]
    Vpred=pphv[index]
    #print(Vpred)
    phpred=omg_inst[-1]*(t_ariv[-1]-dist/Vpred)
    k=(phpred-ph_ariv[-1]+np.pi/4)/np.pi
    k=int((phpred-ph_ariv[-1]+np.pi/4)/np.pi+0.5)
    phase_vel.append(dist/(t_ariv[-1]-(ph_ariv[-1]+k*np.pi-np.pi/4)/omg_inst[-1]))
    for i in range(len(omg_inst)-1):
        i=-(i+2)
        Vpred=phase_vel[-1]
        phpred=omg_inst[i]*(t_ariv[i]-dist/Vpred)
        k=int((phpred-ph_ariv[i]+np.pi/4)/np.pi+0.5)
        phase_vel.append(dist/(t_ariv[i]-(ph_ariv[i]+k*np.pi-np.pi/4)/omg_inst[i]))
    phase_vel.reverse()

    return(phase_vel)

def write_velinfo(ccf_file,NUM,ftan_mat,per_inst,group_vel,phase_vel,SSNR):
    #写 ftan_mat
    ftan_mat=np.transpose(ftan_mat)
    ff=open(ccf_file+'_AMP_'+str(NUM),'w')
    for m in range(len(per_inst)):
        for n in range(len(ftan_mat[m])):
            ff.write(format('%8.3f'%per_inst[m])+' '*4+\
            format('%8.4f'%ftan_mat[m][n])+'\n')
    ff.close()
    #写速度信息
    ff=open(ccf_file+'_DISP_'+str(NUM),'w')
    if NUM==1:
        for m in range(len(per_inst)):
            ff.write(format('%8.3f'%per_inst[m])+' '*4+format('%8.4f'%group_vel[m])\
                     +' '*4+format('%8.4f'%phase_vel[m])+' '*4+\
                     format('%8.4f'%SSNR[m])+'\n')
    elif NUM==2:
        for m in range(len(per_inst)):
            ff.write(format('%8.3f'%per_inst[m])+' '*4+format('%8.4f'%group_vel[m])\
                     +' '*4+format('%8.4f'%phase_vel[m])+'\n')
    ff.close()

    return()

def phase_match_filter(ccf,t_ccf,ccf_full,dist,group_vel,per_inst,phamafactor):
    ccf_full_orig=ccf_full.copy()
    #由第一次计算的结果求群速度曲线的倒数以及对应的频率
    group_vel_recip=1/np.array(group_vel[::-1])
    omg_inst=1/np.array(per_inst[::-1])
    #变换到频率域
    ccf_freq=fftpack.fft(ccf)
    freq_samp=fftpack.fftfreq(len(ccf))
    ccf_full_freq=fftpack.fft(ccf_full)
    freq_full_samp=fftpack.fftfreq(len(ccf_full))
    #对group_vel_recip积分求Kw,进而求相位校正
    phase_modf=[]
    for omgn in freq_samp:
        omgn_abs=abs(omgn)
        if omgn_abs<=omg_inst[0]:
            phase_modf.append(0)
        elif omgn_abs<=omg_inst[-1]:
            multi_omgn=[omgn_abs]*len(omg_inst)
            diff=abs(np.array(omg_inst)-np.array(multi_omgn))
            index=np.where(diff==np.min(diff))[0][0]
            Komgn=np.trapz(group_vel_recip[:index+1],omg_inst[:index+1])
            if omgn<0:
                Komgn=-Komgn
            phase_modf.append(Komgn*dist)
        else:
            phase_modf.append(0)
    phase_full_modf=[]
    for omgn in freq_full_samp:
        omgn_abs=abs(omgn)
        if omgn_abs<=omg_inst[0]:
            phase_full_modf.append(0)
        elif omgn_abs<=omg_inst[-1]:
            multi_omgn=[omgn_abs]*len(omg_inst)
            diff=abs(np.array(omg_inst)-np.array(multi_omgn))
            index=np.where(diff==np.min(diff))[0][0]
            Komgn=np.trapz(group_vel_recip[:index+1],omg_inst[:index+1])
            if omgn<0:
                Komgn=-Komgn
            phase_full_modf.append(Komgn*dist)
        else:
            phase_full_modf.append(0)
    #对corr_freq进行相位匹配滤波并返回到时域
    ccf_freq=ccf_freq*np.exp(1j*np.array(phase_modf))
    ccf=fftpack.ifft(ccf_freq).real
    ccf_hilbert=fftpack.hilbert(ccf)
    ccf_env=np.sqrt(ccf**2+ccf_hilbert**2)
    #去除噪声
    index=np.where(ccf_env==np.max(ccf_env))[0][0]
    #left
    index_left=index
    try:
        OK=0
        while OK==0:
            if ccf_env[index_left-1]<ccf_env[index_left]:
                index_left=index_left-1
            elif ccf_env[index_left]/ccf_env[index]<phamafactor:
                OK=1
            else:
                index_left=index_left-1
    except:
        index_left=0
    #right
    index_right=index
    try:
        OK=0
        while OK==0:
            if ccf_env[index_right+1]<ccf_env[index_right]:
                index_right=index_right+1
            elif ccf_env[index_right]/ccf_env[index]<phamafactor:
                OK=1
            else:
                index_right=index_right+1
    except:
        index_right=len(ccf_env)-1
    for i in range(index_left):
        ccf[i]=0
    for i in range(len(ccf)-index_right):
        ccf[-(i+1)]=0

    #构建清除了干扰信号的 ccf_full
    signal_left=np.zeros(int(t_ccf[0]))
    signal_right=np.zeros(int(len(ccf_full)-t_ccf[-1]-1))
    ccf_full=np.concatenate((signal_left,ccf))
    ccf_full=np.concatenate((ccf_full,signal_right))

    #进行反相位匹配滤波并返回到时域
    ccf_freq=fftpack.fft(ccf)
    ccf_freq=ccf_freq/np.exp(1j*np.array(phase_modf))
    ccf=fftpack.ifft(ccf_freq).real
    ccf_full_freq=fftpack.fft(ccf_full)
    ccf_full_freq=ccf_full_freq/np.exp(1j*np.array(phase_full_modf))
    ccf_full=fftpack.ifft(ccf_full_freq).real

    #plt.figure()
    #plt.plot(ccf_full_orig,'k')
    #plt.plot(ccf_full,'r')

    return(ccf,ccf_full)
