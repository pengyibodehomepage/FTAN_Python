#!/usr/bin/env python3

from read_info import readinfo
from func_lib import pre_treat
from func_lib import write_velinfo
from func_lib import compute_group_vel
from func_lib import compute_phase_vel
from func_lib import phase_match_filter
import matplotlib.pyplot as plt

#读取参数和 CCF 的基本信息
[ccf,ccf_file,dist,samp,vmin,vmax,\
      tmin,tmax,alpha,PHV,phamafactor]=readinfo() 

#对 CCF 进行预处理
[ccf_full,ccf,t_ccf,vel_scale]=pre_treat(ccf,dist,samp,vmin,vmax)

#初步计算群速度
ccf1=ccf.copy()
t_ccf1=t_ccf.copy()
print('\n正在计算群速度...','Phase 1')
[ftan_mat1,group_vel1,per_inst1,omg_inst1,t_ariv1,ph_ariv1,SNR1]=\
compute_group_vel(ccf_full,ccf1,t_ccf1,dist,tmin,tmax,vmin,vmax,alpha)

#初步计算相速度
print('\n正在计算相速度...','Phase 1')
phase_vel1=compute_phase_vel\
(group_vel1,per_inst1,omg_inst1,dist,t_ariv1,ph_ariv1,PHV)

#写入计算结果
write_velinfo(ccf_file,1,ftan_mat1,per_inst1,group_vel1,phase_vel1,SNR1)

#画图
plt.figure()
plt.contourf(per_inst1,vel_scale,ftan_mat1,8,alpha=.85,cmap='jet')
plt.plot(per_inst1,group_vel1,'k',linewidth=2)
plt.plot(per_inst1,phase_vel1,'-k',linewidth=2)
plt.title('BASIC COMPUTATION    distance='+str(round(dist,2))+'KM')
plt.xlabel('T (s)')
plt.ylabel('Vel (KM/s)')

#对 CCF 进行相位匹配滤波
ccf2=ccf.copy()
t_ccf2=t_ccf.copy()
[ccf2,ccf_full2]=phase_match_filter(ccf2,t_ccf2,ccf_full,dist,\
                              group_vel1,per_inst1,phamafactor)

#第二次计算群速度
print('\n正在计算群速度...','Phase 2')
[ftan_mat2,group_vel2,per_inst2,omg_inst2,t_ariv2,ph_ariv2,SNR2]=\
compute_group_vel(ccf_full2,ccf2,t_ccf2,dist,tmin,tmax,vmin,vmax,alpha)

#第二次计算相速度
print('\n正在计算相速度...','Phase 2')
phase_vel2=compute_phase_vel\
(group_vel2,per_inst2,omg_inst2,dist,t_ariv2,ph_ariv2,PHV)

#写入计算结果
write_velinfo(ccf_file,2,ftan_mat2,per_inst2,group_vel2,phase_vel2,SNR2)

#画图
plt.figure()
plt.contourf(per_inst2,vel_scale,ftan_mat2,8,alpha=.85,cmap='jet')
plt.plot(per_inst2,group_vel2,'k',linewidth=2)
plt.plot(per_inst2,phase_vel2,'-k',linewidth=2)
plt.title('PHASE MATCH FILTER COMPUTATION    distance='+str(round(dist,2))+'KM')
plt.xlabel('T (s)')
plt.ylabel('Vel (KM/s)')
plt.show()

