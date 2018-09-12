import tensorflow as tf
import numpy as np
import cv2
import time

PI=3.141592653589793
DBL_MIN=0.01

class EFCHandler:
    def __init__(self):
        self.data=1
    
    def CalculateAssistVariable(self,assist_array_length):
        assist_2npi=[0.0 for i in range(assist_array_length)]
        assist_2_nsq_pisq=[0.0 for i in range(assist_array_length)]
        for n in range(assist_array_length):
            assist_2npi[n]=2*n*PI
            assist_2_nsq_pisq[n]=2*n*n*PI*PI
            
        return assist_2npi,assist_2_nsq_pisq
        
    def PartEncode(self,contour_points,coef_num,_period):
        coef=[[0.0 for i in range(4)] for i in range(coef_num)]
        if len(contour_points)==0:
            return coef
        #Calculate the assist vaiable
       
        assist_2npi,assist_2_nsq_pisq=self.CalculateAssistVariable(coef_num)
        
        point_num=len(contour_points)
        seg_num=point_num-1
        
        deltaX=[0.0 for i in range(seg_num)]
        deltaY=[0.0 for i in range(seg_num)]
        deltaT=[0.0 for i in range(seg_num)]
        
        for i in range(seg_num):
            #contour_points[i][0]:x
            deltaX[i]=contour_points[i+1][0]-contour_points[i][0]
            #contour_points[i][1]:y
            deltaY[i]=contour_points[i+1][1]-contour_points[i][1]
            deltaT[i]=np.sqrt(deltaX[i]*deltaX[i]+deltaY[i]*deltaY[i])

        #remove the repeat point??????????????????
        #for i in range(len(deltaT)):
            #if deltaT[i]<DBL_MIN:
                
        seg_num=len(deltaT)
        point_num=seg_num+1
        
        #calculate time stamp
        time_stamp=[0.0 for i in range(point_num)]
        for i in range(1,point_num):
            time_stamp[i]=time_stamp[i-1]+deltaT[i-1]
            
        period=time_stamp[point_num-1]
        ellipse_dcX=0.0
        
        #Calculate A0
        for i in range(seg_num):
            epsilon=0.0
            for j in range(i):
                epsilon=epsilon+deltaX[j]
            epsilon=epsilon-deltaX[i]/deltaT[i]*time_stamp[i]
            single_item=deltaX[i]/(2.0*deltaT[i])*(time_stamp[i+1]*time_stamp[i+1]-time_stamp[i]*time_stamp[i])+epsilon*(time_stamp[i+1]-time_stamp[i])
            ellipse_dcX=ellipse_dcX+single_item
        #A0    
        coef[0][0]=ellipse_dcX/period+contour_points[0][0]
        
        #Calculate C0
        ellipse_dcY=0.0
        for i in range(seg_num):
            epsilon=0.0
            for j in range(i):
                 epsilon=epsilon+deltaY[j]
            epsilon=epsilon-deltaY[i]/deltaT[i]*time_stamp[i]
            single_item=deltaY[i]/(2.0*deltaT[i])*(time_stamp[i+1]*time_stamp[i+1]-time_stamp[i]*time_stamp[i])+epsilon*(time_stamp[i+1]-time_stamp[i])
            ellipse_dcY=ellipse_dcY+single_item
        #C0                
        coef[0][2]=ellipse_dcY/period+contour_points[0][1]
        
        #Calculate a_n
        for i in range(1,coef_num):
            accum=0.0
            for j in range(seg_num):
                accum=accum+deltaX[j]/deltaT[j]*(np.cos(assist_2npi[i]*time_stamp[j+1]/period)-np.cos(assist_2npi[i]*time_stamp[j]/period))
            #A[1~n]
            coef[i][0]=accum*period/assist_2_nsq_pisq[i]
         
        #Calculate b_n
        for i in range(1,coef_num):
            accum=0.0
            for j in range(seg_num):
                accum=accum+deltaX[j]/deltaT[j]*(np.sin(assist_2npi[i]*time_stamp[j+1]/period)-np.sin(assist_2npi[i]*time_stamp[j]/period))
            #B[1~n]
            coef[i][1]=accum*period/assist_2_nsq_pisq[i]
            
        #Calculate c_n
        for i in range(1,coef_num):
            accum=0.0
            for j in range(seg_num):
                accum=accum+(deltaY[j]/deltaT[j]*(np.cos(assist_2npi[i]*time_stamp[j+1]/period)-np.cos(assist_2npi[i]*time_stamp[j]/period)))
            #A[1~n]
            coef[i][2]=accum*period/assist_2_nsq_pisq[i]
         
        #Calculate d_n
        for i in range(1,coef_num):
            accum=0.0
            for j in range(seg_num):
                accum=accum+(deltaY[j]/deltaT[j]*(np.sin(assist_2npi[i]*time_stamp[j+1]/period)-np.sin(assist_2npi[i]*time_stamp[j]/period)))
            #B[1~n]
            coef[i][3]=accum*period/assist_2_nsq_pisq[i]
  
        return coef
    
    def PartDecode(self,part_num,coefficients,reconstruct_coef_num):
        contour=[[0 for i in range(2)] for i in range(part_num)]
        
        if(reconstruct_coef_num>len(coefficients)):
            return contour
        assist_2npi,assist_2_nsq_pisq=self.CalculateAssistVariable(reconstruct_coef_num)
        #reconstruct
        for i in range(part_num):
            accumX=0.0
            accumY=0.0
            for j in range(1,reconstruct_coef_num):
                accumX=accumX+coefficients[j][0]*np.cos(assist_2npi[j]*i/part_num)+coefficients[j][1]*np.sin(assist_2npi[j]*i/part_num)
                accumY=accumY+coefficients[j][2]*np.cos(assist_2npi[j]*i/part_num)+coefficients[j][3]*np.sin(assist_2npi[j]*i/part_num)
            contour[i][0]=(float)(coefficients[0][0]+accumX)
            contour[i][1]=(float)(coefficients[0][2]+accumY)
            
        return contour
        
    def Part(self,input_points,reconstruction_coef_num,part_num):
        period=1
        coef=self.PartEncode(input_points,reconstruction_coef_num,period)
        output_points=self.PartDecode(part_num,coef,reconstruction_coef_num)
        
        return output_points
    