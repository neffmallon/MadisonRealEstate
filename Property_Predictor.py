# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:38:45 2018

@author: Nathan
"""
### Package Import
# Data Analysis and wrangling
import pandas as pd
import numpy as np
import scipy as sp
import time
import datetime
# visualization
from matplotlib import cm
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.fftpack import fft,ifft

# Machine learning models
from sklearn.linear_model import LogisticRegression

### import data set

tax_file_name = 'cleaned tax parcels.csv'
sales_file_name = 'Sales.csv'

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

### Building Data Set
# Previously I was dropping col. Now I'm only loading what I want
#drop_col = ['OBJECTID', 'Parcel', 'XRefParcel', 'StreetDir', 'StreetName',
#            'StreetType', 'Address','AreaName','RefuseURL','BlockNumber',
#            'AssessedByState']
# All Columns:
#OBJECTID,Parcel,XRefParcel,Address,DateParcelChanged,PropertyClass,PropertyUse,
#AssessmentArea,AreaName,MoreThanOneBuild,HomeStyle,DwellingUnits,Stories,
#YearBuilt,Bedrooms,FullBaths,HalfBaths,TotalLivingArea,FirstFloor,SecondFloor,
#ThirdFloor,AboveThirdFloor,FinishedAttic,Basement,FinishedBasement,
#ExteriorWall1,ExteriorWall2,Fireplaces,CentralAir,PartialAssessed,
#AssessedByState,CurrentLand,CurrentImpr,CurrentTotal,PreviousLand,PreviousImpr
#,PreviousTotal,NetTaxes,SpecialAssmnt,OtherCharges,TotalTaxes,LotSize,Zoning1,
#Zoning2,Zoning3,Zoning4,FrontageFeet,FrontageStreet,WaterFrontage,TIFDistrict,
#TaxSchoolDist,AttendanceSchool,ElementarySchool,MiddleSchool,HighSchool,Ward,
#StateAssemblyDistrict,RefuseDistrict,RefuseURL,PreviousLand2,PreviousImpr2,
#PreviousTotal2,AlderDistrict,AssessmentChangeDate,BlockNumber,BuildingDistrict,
#CapitolFireDistrict,CensusTract,ConditionalUse,CouncilHold,DateAdded,DeedPage,
#DeedRestriction,DeedVolume,ElectricalDistrict,EnvHealthDistrict,ExemptionType,
#FireDistrict,FloodPlain,FuelStorageProximity,HeatingDistrict,Holds,
#IllegalLandDivision,LandfillProximity,LandfillRemediation,Landmark,
#LandscapeBuffer,LocalHistoricalDist,LotDepth,LotNumber,LotteryCredit,LotType1,
#LotType2,LotWidth,MCDCode,NationalHistoricalDist,NeighborhoodDesc,
#NeighborhoodPrimary,NeighborhoodSub,NeighborhoodVuln,NoiseAirport,
#NoiseRailroad,NoiseStreet,ObsoleteDate,OwnerChangeDate,OwnerOccupied,
#ParcelChangeDate,ParcelCode,ParkProximity,Pending,PlanningDistrict,
#PlumbingDistrict,PoliceDistrict,PoliceSector,PreviousClass,PropertyUseCode,
#RailroadFrontage,ReasonChangeImpr,ReasonChangeLand,SenateDistrict,
#SupervisorDistrict,TifImpr,TifLand,TifYear,TotalDwellingUnits,
#TrafficAnalysisZone,TypeWaterFrontage,UWPolice,WetlandInfo,ZoningAll,
#ZoningBoardAppeal,UrbanDesignDistrict,HouseNbr,StreetDir,StreetName,StreetType,
#Unit,StreetID,StormOutfall,FireDemandZone,FireDemandSubZone,PropertyChangeDate,    
#MaxConstructionYear,XCoord,YCoord,SHAPESTArea,SHAPESTLength

### Helper functions for low pass

def Pass_filter(data,cutoff_index=17):
    ft = fft(data)
    low_freq = ft.copy()
    high_freq = ft.copy()
    high_freq[:cutoff_index] = 0
    high_freq[1-cutoff_index:] = 0
    low_freq[cutoff_index:-cutoff_index] = 0
    return ifft(low_freq),ifft(high_freq)

#columns = ['Parcel','TotalLivingArea','Bedrooms','FullBaths','HalfBaths']
#consider = ['YearBuilt','MaxContructionYear','ElementarySchool']

# All columns
'''
columns = ['OBJECTID','Parcel','XRefParcel','Address','DateParcelChanged',
           'PropertyClass','PropertyUse','AssessmentArea','AreaName',
           'MoreThanOneBuild','HomeStyle','DwellingUnits','Stories',
           'YearBuilt','Bedrooms','FullBaths','HalfBaths','TotalLivingArea',
           'FirstFloor','SecondFloor','ThirdFloor','AboveThirdFloor',
           'FinishedAttic','Basement','FinishedBasement','ExteriorWall1',
           'ExteriorWall2','Fireplaces','CentralAir','PartialAssessed',
           'AssessedByState','OtherCharges','LotSize','Zoning1','Zoning2',
           'Zoning3','Zoning4','FrontageFeet','FrontageStreet','WaterFrontage',
           'TIFDistrict','TaxSchoolDist','AttendanceSchool','ElementarySchool',
           'MiddleSchool','HighSchool','Ward','StateAssemblyDistrict',
           'RefuseDistrict','RefuseURL','AlderDistrict','AssessmentChangeDate',
           'BlockNumber','BuildingDistrict','CapitolFireDistrict','CensusTract',
           'ConditionalUse','CouncilHold','DateAdded','DeedPage','DeedRestriction',
           'DeedVolume','ElectricalDistrict','EnvHealthDistrict','ExemptionType',
           'FireDistrict','FloodPlain','FuelStorageProximity','HeatingDistrict',
           'Holds','IllegalLandDivision','LandfillProximity','LandfillRemediation',
           'Landmark','LandscapeBuffer','LocalHistoricalDist','LotDepth',
           'LotNumber','LotteryCredit','LotType1','LotType2','LotWidth',
           'MCDCode','NationalHistoricalDist','NeighborhoodDesc',
           'NeighborhoodPrimary','NeighborhoodSub','NeighborhoodVuln',
           'NoiseAirport','NoiseRailroad','NoiseStreet','ObsoleteDate',
           'OwnerChangeDate','OwnerOccupied','ParcelChangeDate','ParcelCode',
           'ParkProximity','Pending','PlanningDistrict','PlumbingDistrict',
           'PoliceDistrict','PoliceSector','PreviousClass','PropertyUseCode',
           'RailroadFrontage','ReasonChangeImpr','ReasonChangeLand',
           'SenateDistrict','SupervisorDistrict','TifImpr','TifLand','TifYear',
           'TotalDwellingUnits','TrafficAnalysisZone','TypeWaterFrontage',
           'UWPolice','WetlandInfo','ZoningAll','ZoningBoardAppeal',
           'UrbanDesignDistrict','HouseNbr','StreetDir','StreetName',
           'StreetType','Unit','StreetID','StormOutfall','FireDemandZone',
           'FireDemandSubZone','PropertyChangeDate','MaxConstructionYear',
           'XCoord','YCoord','SHAPESTArea','SHAPESTLength']
'''
           # Identifiers
columns = ['OBJECTID','Parcel','XRefParcel','Address','Unit',
           
           'DateParcelChanged',
           'PropertyClass','PropertyUse',
           # Assessor Values:
           'CurrentLand','CurrentImpr','CurrentTotal','PreviousLand',
           'PreviousImpr','PreviousTotal',
           # Geographical divisions
           'AssessmentArea','AreaName','Ward','RefuseDistrict','AlderDistrict',
           'CensusTract','PoliceSector','TrafficAnalysisZone',
           # Schools
           'AttendanceSchool','ElementarySchool','MiddleSchool','HighSchool',
           # Building characteristics
           'MoreThanOneBuild','HomeStyle','DwellingUnits','TotalDwellingUnits',
           'Stories','YearBuilt','MaxConstructionYear',
           'Bedrooms','FullBaths','HalfBaths','TotalLivingArea',
           'FirstFloor','SecondFloor','ThirdFloor','AboveThirdFloor',
           'FinishedAttic','Basement','FinishedBasement','ExteriorWall1',
           'ExteriorWall2','Fireplaces','CentralAir','Landmark',
           # lot characteristics
           'LotNumber','LotType1','LotType2','LotSize','LotWidth','LotDepth',
           'FrontageFeet','FrontageStreet','WaterFrontage','TypeWaterFrontage',
           'RailroadFrontage','WetlandInfo',
           # ???
           'PartialAssessed','AssessedByState','OtherCharges','AssessmentChangeDate',
           # Location characteristics
           'FloodPlain','FuelStorageProximity','LandfillProximity',
           'LandfillRemediation','LocalHistoricalDist','NationalHistoricalDist',
           'NoiseAirport','NoiseRailroad','NoiseStreet',
           # Various questionable use columns
           'ConditionalUse','CouncilHold','DateAdded','DeedPage','DeedRestriction',
           'DeedVolume','ExemptionType','Holds',
           'LandscapeBuffer','LotteryCredit','MCDCode','ObsoleteDate',
           'OwnerChangeDate','OwnerOccupied','ParcelChangeDate','ParcelCode',
           'ParkProximity','Pending','PreviousClass','ReasonChangeImpr',
           'ReasonChangeLand','TifImpr','TifLand','TifYear',
           'StormOutfall','PropertyChangeDate',
           # Plotting
           'XCoord','YCoord','SHAPESTArea','SHAPESTLength']

districts = ['AssessmentArea','AreaName','Ward','RefuseDistrict','AlderDistrict',
             'CensusTract','PoliceSector','TrafficAnalysisZone']

useful_districts = ['ElementarySchool','PoliceSector','CensusTract',
                    'TrafficAnalysisZone', 'Ward']

borin_disctricts = ['TIFDistrict','TaxSchoolDist','BuildingDistrict',
                    'CapitolFireDistrict','ElectricalDistrict',
                    'EnvHealthDistrict','FireDistrict','HeatingDistrict',
                    'NeighborhoodDesc','NeighborhoodPrimary','NeighborhoodSub',
                    'NeighborhoodVuln','PlanningDistrict','PlumbingDistrict',
                    'PoliceDistrict','SenateDistrict',
                    'SupervisorDistrict','UrbanDesignDistrict',
                    'FireDemandZone','UWPolice']
districts.sort()

def load_n_merge():
    A = time.time()
    df_tax = pd.read_csv(tax_file_name, low_memory=False, index_col = 1,
                         dtype = {'Parcel' : np.int64},
                         usecols = columns)
    df_sales = pd.read_csv(sales_file_name,
                           dtype = {'Price' : np.int64,'Parcel' : np.int64},
                           parse_dates = ['Date'])
    df_sales = df_sales[(df_sales.Date < datetime.datetime.strptime('2018-01-01 00:00:00','%Y-%m-%d %H:%M:%S'))]
    df_sales = pd.merge(df_sales, df_tax, left_on = 'Parcel', right_index=True)
    df_sales = df_sales[(df_sales.TotalLivingArea>0)]
    print('File Loaded\n--- {} seconds ---'.format((time.time()-A)))
    return df_tax, df_sales

#dt_tax = dt_tax.drop(drop_col, axis = 1)


### Main Exicution

if __name__ == '__main__':
    if True:
        try:
            del df_tax
            del df_sales
        except:
            pass
        df_tax,df_sales = load_n_merge()
        corrmat = df_sales.corr()
        #f, ax = plt.subplots(figsize=(12, 9))
        #sns.heatmap(corrmat, vmax=.8, square=True)
        
    if True:
        for d_name in districts:
            plt.figure(figsize=(8, 6), dpi=250)
            cats = df_tax[d_name].astype('category').values.categories
            legend = []
            for cat in cats:
                X = df_tax[(df_tax[d_name] == cat)]['XCoord']
                Y = df_tax[(df_tax[d_name] == cat)]['YCoord']
                plt.scatter(X,Y)
                legend.append(cat)
            plt.legend(legend, fontsize = 6, ncol=2)
            plt.title(d_name)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(d_name + '.png')
            plt.close()
            
        
        
        
    if False: ###Price correlation matrix
        corrmat = df_sales.corr()
        k = 10 #number of variables for heatmap
        cols = corrmat.nlargest(k, 'Price')['Price'].index
        corr_mat = np.corrcoef(df_sales[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(corr_mat, cbar=True, annot=True, square=True,
                         fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
    if False:
        A = time.time()
        day_one = df_sales.Date.min().value
        date_value = lambda x : (x.value - day_one)/(3600*24*1000000000)
        df_sales['Date_value'] = df_sales['Date'].map(date_value)
        del date_value
        df_sales['Month'] = df_sales['Date'].map(lambda x: x.month)
        df_sales['Year'] = df_sales['Date'].map(lambda x: x.year)
        print('Dates Processed\n--- %s seconds ---' %(time.time()-A))
        A=time.time()
        df_homes = df_sales[(df_sales.TotalLivingArea>0)]
        df_homes = df_homes[(df_homes.PropertyUse == 'Single family')]
        df_homes['Price_per_Sqft'] = df_homes.Price / df_homes.TotalLivingArea
        print('price/sqft Processed\n--- %s seconds ---' %(time.time()-A))
    if False:
        # calculate mean ppsf grouped by month
        dates = df_homes.Date.astype('category').values.categories
        dates = [d for d in dates]
        means_ppsf = [df_homes[(df_homes.Date == d)]['Price_per_Sqft'].mean() for d in dates]
        stdev_ppsf = [df_homes[(df_homes.Date == d)]['Price_per_Sqft'].std() for d in dates]
        # caculate a spline
        date_values = df_homes.Date_value.astype('category').values.categories
        price_vs_time_spline = sp.interpolate.UnivariateSpline(date_values,means_ppsf, s=5000.0)
        # calculate a rolling average of the price
        price_vs_time_rolling_ave = running_mean(means_ppsf,12)
        dev_from_mean = means_ppsf[5:-6]-price_vs_time_rolling_ave
        fft_dev_from_means = sp.fftpack.fft(dev_from_mean)
        # plot
        plt.errorbar(date_values/30, means_ppsf, stdev_ppsf)
        
        plt.plot(date_values/30, price_vs_time_spline(date_values),
                 date_values[5:-6]/30, price_vs_time_rolling_ave)
        
        plt.xlabel('Months since January 2002')
        plt.ylabel("Price per sq. ft.")
        plt.legend(["12-month rolling average","Montly mean","Spline fit"])
        
    if False:
        # Goal: Make a plot of the price per square foot of madison
        # in deviations from the mean
        mean_lookup = lambda date: means_ppsf[dates.index(date)]
        stdev_lookup = lambda date: stdev_ppsf[dates.index(date)]
        
        df_homes['Monthly_Mean'] = df_homes.Date.map(mean_lookup)
        df_homes['Monthly_Stdev'] = df_homes.Date.map(stdev_lookup)
        df_homes['Dev_from_mean'] = (df_homes['Price_per_Sqft'] - df_homes['Monthly_Mean'])/df_homes['Monthly_Stdev']
        
        #plt.figure(num=None, figsize=(8, 6), dpi=250, facecolor='w', edgecolor='k')
        #plt.tricontourf(df_homes.XCoord, df_homes.YCoord,df_homes['Dev_from_mean'],10)
        #plt.savefig('Madison_map.png')
            
        neighborhoods = df_homes.ElementarySchool.astype('category').values.categories
        neighborhood_means = [df_homes[(df_homes.ElementarySchool == n)]['Dev_from_mean'].mean() for n in neighborhoods]
        elems = [(mean,n) for n, mean in zip(neighborhoods,neighborhood_means)]
        elems.sort()
        # plot elementary school zones
        #df_homes['Elem_ppsf'] = df_homes['Dev_from_mean']
        plt.figure(figsize=(8, 6), dpi=250)
        legend = []
        # set up colors
        cmap = cm.seismic
        limit = max(abs(elems[0][0]),abs(elems[-1][0]))
        cmap_adj = lambda zscore: (-zscore + limit)/(2*limit)
        # plot all the schools
        for zscore, school in elems:
            adj_zscore = cmap_adj(zscore)
            X = df_homes[(df_homes.ElementarySchool == school)]['XCoord']
            Y = df_homes[(df_homes.ElementarySchool == school)]['YCoord']
            plt.scatter(X,Y,c=cmap(adj_zscore), marker='.', alpha = .1)#, cmap=cmap)
            legend.append(school)
            
        plt.legend(legend, fontsize = 6, ncol=2)
        plt.title("Neighborhood Deviation from Madison Average Single-Family Price per Square Foot")
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()

    if False:
        # Goal: Make a plot of the price per square foot of madison
        # in deviations from the mean
        mean_lookup = lambda date: means_ppsf[dates.index(date)]
        stdev_lookup = lambda date: stdev_ppsf[dates.index(date)]
        
        df_homes['Monthly_Mean'] = df_homes.Date.map(mean_lookup)
        df_homes['Monthly_Stdev'] = df_homes.Date.map(stdev_lookup)
        df_homes['Dev_from_mean'] = (df_homes['Price_per_Sqft'] - df_homes['Monthly_Mean'])/df_homes['Monthly_Stdev']
        
        #plt.figure(num=None, figsize=(8, 6), dpi=250, facecolor='w', edgecolor='k')
        #plt.tricontourf(df_homes.XCoord, df_homes.YCoord,df_homes['Dev_from_mean'],10)
        #plt.savefig('Madison_map.png')
            
        neighborhoods = df_homes.TrafficAnalysisZone.astype('category').values.categories        
        neighborhood_means = [df_homes[(df_homes.TrafficAnalysisZone == n)]['Dev_from_mean'].mean() for n in neighborhoods]
        elements = [(mean,n) for n, mean in zip(neighborhoods,neighborhood_means)]
        elements.sort()
        plt.figure(figsize=(8, 6), dpi=250)
        legend = []
        # set up colors
        cmap = cm.seismic
        limit = max(abs(elements[0][0]),abs(2))
        cmap_adj = lambda zscore: (-zscore + limit)/(2*limit)
        # plot all the schools
        for zscore, zone in elements:
            adj_zscore = cmap_adj(zscore)
            X = df_homes[(df_homes.TrafficAnalysisZone == zone)]['XCoord']
            Y = df_homes[(df_homes.TrafficAnalysisZone == zone)]['YCoord']
            plt.scatter(X,Y,c=cmap(adj_zscore), marker='.')#, cmap=cmap)
            
        plt.legend(legend, fontsize = 6, ncol=2)
        plt.title("Neighborhood Deviation from Madison Average Single-Family Price per Square Foot")
        plt.xticks([])
        plt.yticks([])
                

    if False:
        ### Category Indexes
        # Category lists can be made by using astype('category')
        # Example
        TWF_list = df_tax.TypeWaterFrontage.astype('category').values.categories
        TWF_lib = {}
        for i,s in enumerate(TWF_list):
            TWF_lib[s] = i
        df_tax['TypeWaterFrontage']=df_tax['TypeWaterFrontage'].map(TWF_lib).astype(int)
        # you can make this a dictionary with enumerate or some shit, idk, I'm tired
        
        ### Turn String category shit into numerical category shit
        
        df_tax['CentralAir'] = df_tax['CentralAir'].map( {'YES':1, 'NO':0} ).astype(int)
        df_tax.loc[df_tax.MoreThanOneBuild.isnull(), 'MoreThanOneBuild'] = '0'
        df_tax['MoreThanOneBuild'] = df_tax['MoreThanOneBuild'].map( {'Has more than one building':1, '0':0} ).astype(int)
        df_tax['PropertyClass']=df_tax['PropertyClass'].map( {'Agricultural':0, 'Residential':1} ).astype(int)
        df_tax.loc[df_tax.Railroadfrontage.isnull(),'Railroadfrontage'] = '0'
        df_tax['Railroadfrontage']=df_tax['Railroadfrontage'].map( {'Railroad Frontage':1,'0':0} ).astype(int)