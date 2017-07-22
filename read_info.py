#!/usr/bin/env python3

import os
import numpy as np
from obspy import read

def readinfo():
    '''
    此函数直接读取了参数文件中的参数,并依据参数文件中的 CCF 文件名(
    含相对路径)读取了 CCF ,并读取了 CCF 的基本信息,之后将必要信息返
    回.

    Return
    ------
    tr : Obspy.Trace
        互相关函数
    dist: float
        台站间距
    samp: float
        互相关函数采样率
    vmin,vmax,tmin,tmax: float
        详见 README
    '''
    #读取参数文件
    PFN='param_dat'    #param_file_name
    if not os.path.exists(PFN):
        msg=PFN+' not exist!'
        raise Exception(msg)

    ff=open(PFN)
    param=ff.readlines()
    ff.close()

    tmp=[]
    for pp in param:
        if pp[0]!='#': tmp.append(pp[:-1])

    #读取参数
    vmin=tmp[0]
    vmax=tmp[1]
    tmin=tmp[2]
    tmax=tmp[3]
    CCF=tmp[4]
    phamafactor=tmp[5]
    STA_st=CCF[:-4].split('_')[-2]
    STA_ev=CCF[:-4].split('_')[-1]
    #print(STA_st,STA_ev)

    #读取互相关函数信息
    tr=read(CCF)[0]
    samp=tr.stats.sampling_rate
    stlo=tr.stats.sac.stlo
    evlo=tr.stats.sac.evlo
    stla=tr.stats.sac.stla
    evla=tr.stats.sac.evla
    dist=tr.stats.sac.dist
    #确定 Alpha ,参考瑞兹五勒
    alpha=20*np.sqrt(float(dist)/1000)

    #读取平均相速度值
    PHV='PHV_dat'    #phase_velocity_file_name
    if not os.path.exists(PHV):
        msg=PHV+' not exist!'
        raise Exception(msg)
    PHV=np.loadtxt(PHV)

    #打印基本信息
    print('1.','台站 1 :',STA_st,'台站 2 :',STA_ev)
    print('2.','台站 1 经纬度:',stlo,stla,'台站 2 经纬度:',evlo,evla)
    print('3.','台站间距:',dist,'KM',' CCF 采样率:',samp,'HZ')
    #(注:台站 1 , 2 与 CCF 文件名中两个台站出现的次序对应)

    #打印参数信息
    print('4.','vmin =',vmin,'\t','vmax =',vmax,'\t','KM/s')
    print('5.','tmin =',tmin,'\t','tmax =',tmax,'\t','s')

    return(tr,CCF,STA_st,STA_ev,float(dist),float(samp),float(vmin),\
           float(vmax),float(tmin),float(tmax),alpha,PHV,float(phamafactor))
