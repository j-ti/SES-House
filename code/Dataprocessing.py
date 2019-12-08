import pandas as pd
import numpy as np
import os,os.path
import csv
import re
from dateutil.parser import parse
import datetime

#load data

def DataLoadIn():
    LoadFile=pd.read_csv("../sample/pecan-home2233-grid_solar-20190602.csv",header=0,names=['DateTime','Grid Power','Solar Power'])
    LoadFile['DateTime']=LoadFile['DateTime'].apply(DeleteLastTwo)
    LoadFile['Grid Power']=LoadFile['Grid Power'].apply(DeleteLastTwo)
    LoadFile['DateTime']=pd.to_datetime(LoadFile['DateTime'])

    LoadFile.fillna(method='bfill',inplace=True)
    LoadFile=LoadFile.astype({'Grid Power': np.int,'Solar Power':np.int})
    LoadFile['Load']=LoadFile['Grid Power']+LoadFile['Solar Power']
    LoadFile=LoadFile.drop(columns=['Grid Power','Solar Power'])
    #Pv data
    PVFile=pd.read_csv("../sample/ninja_pv_52.5170_13.3889_corrected.csv",header=3,usecols=[0,2],names=['DateTime','Solar Power'])
    PVFile=PVFile.astype({'Solar Power': np.int})
    PVFile['DateTime']=pd.to_datetime(PVFile['DateTime'])

    #Wind data
    WindFile=pd.read_csv("../sample/ninja_wind_52.5170_13.3889_corrected.csv",header=3, usecols=[0,2],names=['DateTime','Wind Power'])
    WindFile=WindFile.astype({'Wind Power': np.int})
    WindFile['DateTime']=pd.to_datetime(WindFile['DateTime'])
    #Price data
    PriceFile=pd.read_csv("../sample/pecan-iso_neiso-day_ahead_lmp_avg-20190602.csv",header=0,names=['DateTime','Electricity Price'])
    PriceFile=PriceFile.astype({'Electricity Price': np.int})
    PriceFile['DateTime']=PriceFile['DateTime'].apply(DeleteLastTwo)
    PriceFile['DateTime']=pd.to_datetime(PriceFile['DateTime'])
    return [LoadFile,PVFile,WindFile,PriceFile]
#function for loading data
def DeleteLastTwo(string):
    index=string.find(';')
    return string[:index]

def MatrixOutput(LoadFile,PVFile,WindFile,PriceFile,TimeStepDic,StartTime,EndTime,DefinedTimeStep):
    LoadFile=ExtractDatafromDate(LoadFile,StartTime,EndTime)
    PVFile = ExtractDatafromDate(PVFile, StartTime, EndTime)
    PriceFile = ExtractDatafromDate(PriceFile, StartTime, EndTime)
    WindFile=ExtractDatafromDate(WindFile, StartTime, EndTime)
    print(PVFile)
    LoadFile=TimeMap(LoadFile,TimeStepDic["Load"],DefinedTimeStep)
    PVFile = TimeMap(PVFile, TimeStepDic["Solar Power"], DefinedTimeStep)
    WindFile = TimeMap(WindFile, TimeStepDic["Wind Power"], DefinedTimeStep)
    PriceFile = TimeMap(PriceFile, TimeStepDic["Price"], DefinedTimeStep)
    return [LoadFile,PVFile,WindFile,PriceFile]


def ExtractDatafromDate(File,StartTime,EndTime):
    a=StartTime<=File['DateTime']
    b=EndTime>=File['DateTime']
    return File.loc[a & b]
def TimeMap(File,TimeStep,DefinedTimeStep):
    SumIndex=DefinedTimeStep/TimeStep
    DataName=File.columns[1]
    d = {'DateTime': 'first',DataName:'sum'}
    File=File.reset_index()
    ResultFile= File.groupby(File.index // SumIndex).agg(d)
    return ResultFile



def main():
    [LoadFile, PVFile, WindFile, PriceFile] = DataLoadIn()

    TimeStepDic = {"Load": 1, "Price": 60, "Solar Power": 60, "Wind Power": 60}
    StartTime = pd.to_datetime("2014-05-10 00:00:00")
    EndTime = pd.to_datetime("2014-05-10 23:59:00")
    DefinedTimeStep = 60
    [LoadFile, PVFile, WindFile, PriceFile]= MatrixOutput(LoadFile,PVFile,WindFile,PriceFile,TimeStepDic,\
                                                          StartTime,EndTime,DefinedTimeStep)
    Matrix=pd.merge(LoadFile,PriceFile)
    Matrix=pd.merge(Matrix,PVFile)
    Matrix=pd.merge(Matrix,WindFile)

    return Matrix

Matrix=main()
print(Matrix)