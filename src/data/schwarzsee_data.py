import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import scipy.integrate as spi
import math

#settings
start_date=datetime(2019,1,29,16)
end_date=datetime(2019,3,10,18)

# read files
df_in=(
    pd.read_csv('./AWS/Schwarzsee_2018.txt',header=None,encoding='latin-1',skiprows=7, sep='\\s+',
    names=['Date','Time', 'Discharge', 'Wind Direction','Wind Speed','Maximum Wind Speed', 'Temperature', 'Humidity', 'Pressure', 'Pluviometer'])
)

# Drop
df_in=df_in.drop(['Pluviometer'],axis=1)

#Add Radiation data
df_in2=pd.read_csv('./AWS/plaffeien.txt',sep='\\s+',skiprows=2)
df_in2['When']=pd.to_datetime(df_in2['time'], format='%Y%m%d%H%M')
df_in2['ods000z0']=pd.to_numeric(df_in2['ods000z0'],errors='coerce')
df_in2['gre000z0']=pd.to_numeric(df_in2['gre000z0'],errors='coerce')
df_in2['Rad'] = df_in2['gre000z0'] - df_in2['ods000z0']
df_in2['DRad'] = df_in2['ods000z0']

# Add Precipitation data
df_in2['Prec']=pd.to_numeric(df_in2['rre150z0'],errors='coerce')
df_in2['Prec']=df_in2['Prec']/2 #5 minute sum
df_in2 = df_in2.set_index('When').resample('5T').ffill().reset_index()

# Datetime
df_in['When']=pd.to_datetime(df_in['Date'] + ' ' + df_in['Time'])
df_in['When']=pd.to_datetime(df_in['When'], format='%Y.%m.%d %H:%M:%S')


# Correct data errors
i = 1
while( df_in.loc[i,'When'] != datetime(2019,2,6,16,15)) :
    if str(df_in.loc[i,'When'].year) !='2019':
        df_in.loc[i,'When']=df_in.loc[i-1,'When']+ pd.Timedelta(minutes=5)
    i = i + 1

while( df_in.loc[i,'When'] != datetime(2019,3,2,15)) :
    if str(df_in.loc[i,'When'].year) !='2019':
        df_in.loc[i,'When']=df_in.loc[i-1,'When']+ pd.Timedelta(minutes=5)
    i = i + 1

while( df_in.loc[i,'When'] != datetime(2019,3,6,16,25)) :
    if str(df_in.loc[i,'When'].year) !='2019':
        df_in.loc[i,'When']=df_in.loc[i-1,'When']+ pd.Timedelta(minutes=5)
    i = i + 1

df_in=df_in.resample('5Min', on='When').first().drop('When', 1).reset_index()

# Fill missing data
for i in range(1, df_in.shape[0]):
    if np.isnan(df_in.loc[i,'Temperature']):
        df_in.loc[i,'Temperature'] = df_in.loc[i-288, 'Temperature' ]
        df_in.loc[i,'Humidity'] = df_in.loc[i-288, 'Humidity' ]
        df_in.loc[i,'Wind Speed'] = df_in.loc[i-288, 'Wind Speed' ]
        df_in.loc[i,'Maximum Wind Speed'] = df_in.loc[i-288, 'Maximum Wind Speed' ]
        df_in.loc[i,'Wind Direction'] = df_in.loc[i-288, 'Wind Direction' ]
        df_in.loc[i,'Pressure'] = df_in.loc[i-288, 'Pressure' ]
        df_in.loc[i,'Discharge'] = df_in.loc[i-288, 'Discharge' ]


mask=(df_in['When'] >= start_date) & (df_in['When'] <= end_date)
df_in=df_in.loc[mask]
df_in=df_in.reset_index()

mask=(df_in2['When'] >= start_date) & (df_in2['When'] <= end_date)
df_in2=df_in2.loc[mask]
df_in2=df_in2.reset_index()

days= pd.date_range(start=start_date, end=end_date, freq='5T')
days = pd.DataFrame({'When': days})

df=pd.merge(days,df_in[['When', 'Discharge','Wind Speed','Maximum Wind Speed','Wind Direction', 'Temperature', 'Humidity', 'Pressure']],on='When')

#Add Radiation DataFrame
df['Rad'] = df_in2['Rad']
df['DRad'] = df_in2['DRad']
df['Prec'] = df_in2['Prec']/1000

mask=(df['When'] >= start_date) & (df['When'] <= end_date)
df = df.loc[mask]
df = df.reset_index()

for i in range(1,df.shape[0]):
    if np.isnan(df.loc[i,'Rad']):
        df.loc[i,'Rad']=df.loc[i-1,'Rad']
    if np.isnan(df.loc[i,'DRad']):
        df.loc[i,'DRad']=df.loc[i-1,'DRad']
    if np.isnan(df.loc[i,'Prec']):
        df.loc[i,'Prec']=df.loc[i-1,'Prec']


df['Fountain']=0 # Fountain run time

df_nights=pd.read_csv('./field/freeze_nights_times.txt',sep='\\s+')

df_nights['Start']=pd.to_datetime(df_nights['Date'] + ' ' + df_nights['start'])
df_nights['End']=pd.to_datetime(df_nights['Date'] + ' ' + df_nights['end'])
df_nights['Start']=pd.to_datetime(df_nights['Start'], format='%Y-%m-%d %H:%M:%S')
df_nights['End']=pd.to_datetime(df_nights['End'], format='%Y-%m-%d %H:%M:%S')

df_nights['Date']=pd.to_datetime(df_nights['Date'], format='%Y-%m-%d')
mask=(df_nights['Date'] >= start_date) & (df_nights['Date'] <= end_date)
df_nights=df_nights.loc[mask]
df_nights=df_nights.reset_index()

for i in range(0, df_nights.shape[0]):
    df_nights.loc[i,'Start']=df_nights.loc[i,'Start']- pd.Timedelta(days=1)
    df.loc[(df['When'] >= df_nights.loc[i,'Start']) & (df['When'] <= df_nights.loc[i,'End']),'Fountain']=1

#Volume used
df['Used'] = df.Fountain * (df.Discharge *5)/1000
for i in range(0, df.shape[0]):
    ts = pd.Series(df.Used[:i])
    v=spi.trapz(ts.values)
    df.loc[i,'Volume']=v

df.Discharge = df.Discharge * df.Fountain

# v_a mean
v_a = df['Wind Speed'].replace(0, np.NaN).mean() # m/s Average Humidity
print(v_a)
df['Wind Speed'] = df['Wind Speed'].replace(0, v_a)

print(df[df['Wind Speed'] == 0])

#CSV output
df.rename(columns={'Wind Speed': 'v_a','Temperature':'T_a','Humidity':'RH','Volume':'V', 'Pressure':'p_a'}, inplace=True)
df_out = df[['When','T_a', 'RH' , 'v_a','Discharge', 'Rad', 'DRad', 'Prec', 'p_a']]
df_out=df_out.round(5)
df_out.to_csv('./output_data/schwarzsee_model_input.csv', sep=',')


# Plots
filename = "./output_data/all_data" + str(end_date.day) + '.pdf'
pp  =  PdfPages(filename)

x=df['When']
y1=df['T_a']
y2=df['Prec']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'k-',linewidth=0.5)
ax1.set_ylabel('Temperature[C]')
ax1.set_xlabel('Days')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'b-',linewidth=0.5)
ax2.set_ylabel('Prec[m]', color='b')
for tl in ax2.get_yticklabels():
    tl.set_color('b')

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

y1=df['V']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'k-')
ax1.set_ylabel('Volume Sprayed[$m^3$]')
ax1.set_xlabel('Days')

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

y1=df['v_a']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'k-',linewidth=0.5)
ax1.set_ylabel('Wind Speed[$ms^{-1}$]')
ax1.set_xlabel('Days')

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

# directions = np.arange(0, 360, 15)
# fig = wind_rose(rose1, directions)
#
# pp.savefig(bbox_inches = "tight")
# plt.clf()

y1=df['p_a']
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'k-',linewidth=0.5)
ax1.set_ylabel('Pressure[hPa]')
ax1.set_xlabel('Days')

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

y1=df['Rad']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'k-',linewidth=0.5)
ax1.set_ylabel('Short Wave Radiation[$Wm^{-2}$]')
ax1.set_xlabel('Days')

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

y1=df['Discharge']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'k-',linewidth=0.5)
ax1.set_ylabel('Discharge[$lmin^{-1}$]')
ax1.set_xlabel('Days')

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

pp.close()


# Plots
filename = "./output_data/data" + '.pdf'
pp  =  PdfPages(filename)

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1, sharex='col', sharey='row', figsize=(15,10))

# fig.suptitle("Field Data", fontsize=14)
x = df.When

y1 = df.T_a
ax1.plot(x, y1, 'k-', linewidth=0.5)
ax1.set_ylabel('T[$C$]')
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

y2 = df.Discharge
ax2.plot(x, y2, 'k-', linewidth = 0.5)
ax2.set_ylabel('Discharge[$kg$]')
ax2.grid()

y3 = df.Rad
ax3.plot(x, y3,'k-', linewidth = 0.5)
ax3.set_ylabel('SWR[$Wm^{-2}$]')
ax3.set_ylim([0,600])
ax3.grid()

ax3t=ax3.twinx()
ax3t.plot(x, df.DRad, 'b-', linewidth = 0.5)
ax3t.set_ylabel('Diffused[$Wm^{-2}$]', color='b')
ax3t.set_ylim([0,600])
for tl in ax3t.get_yticklabels():
    tl.set_color('b')

y4 = df.Prec * 1000
ax4.plot(x, y4,'k-', linewidth = 0.5)
ax4.set_ylabel('Ppt[$mm$]')
ax4.grid()

y5 = df.p_a
ax5.plot(x, y5,'k-', linewidth = 0.5)
ax5.set_ylabel('Pressure[$hpa$]')
ax5.grid()

y6 = df.v_a
ax6.plot(x, y6,'k-', linewidth = 0.5)
ax6.set_ylabel('Wind[$ms^{-1}$]')
ax6.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y1 = df.T_a
ax1.plot(x, y1, 'k-', linewidth=0.5)
ax1.set_ylabel('T[$C$]')
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y2 = df.Discharge * 5
ax1.plot(x, y2, 'k-', linewidth = 0.5)
ax1.set_ylabel('Discharge[$kg$]')
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y3 = df.Rad
ax1.plot(x, y3,'k-', linewidth = 0.5)
ax1.set_ylabel('SWR[$Wm^{-2}$]')
ax1.set_ylim([0,600])
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y4 = df.Prec * 1000
ax1.plot(x, y4,'k-', linewidth = 0.5)
ax1.set_ylabel('Ppt[$mm$]')
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y5 = df.p_a
ax1.plot(x, y5,'k-', linewidth = 0.5)
ax1.set_ylabel('Pressure[$hpa$]')
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y6 = df.v_a
ax1.plot(x, y6,'k-', linewidth = 0.5)
ax1.set_ylabel('Wind[$ms^{-1}$]')
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches = "tight")
plt.clf()

pp.close()
