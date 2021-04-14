

#Lib for Streamlit
# Copyright(c) 2021 - AilluminateX LLC
# This is main Sofware... Screening and Tirage
# Customized to general Major Activities
# Make all the School Activities- st.write(DataFrame) ==> (outputs) Commented...
# The reason, since still we need the major calculations.
# Also the Computing is not that expensive.. So, no need to optimize at this point

import streamlit as st
import pandas as pd



#Change website title  (set_page_config)
#==============
from PIL import Image

image_favicon=Image.open('Logo_AiX.jpg') 
st.set_page_config(page_title='AilluminateX - Covid Platform', page_icon = 'Logo_AiX.jpg') #, layout = 'wide', initial_sidebar_state = 'auto'), # layout = 'wide',)

# favicon being an object of the same kind as the one you should provide st.image() with 
#(ie. a PIL array for example) or a string (url or local file path)

#==============


#Hide footer and customize the text
#=========================
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                    content:'Copyright(c) 2021 - AilluminateX LLC and Ailysium - Covid19 Bio-Forecasting Platform | https://www.aillumiante.com'; 
                    visibility: visible;
                    display: block;
                    position: relative;
                    #background-color: gray;
                    padding: 5px;
                    top: 2px;
                        }
            
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

#==============================


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import accuracy_score
#import numpy as np
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import altair as alt
import plotly.figure_factory as ff
import matplotlib
from matplotlib import cm
import seaborn as sns; sns.set()
from PIL import Image


import statsmodels.api as sm
import statsmodels.formula.api as smf




#from sklearn import model_selection, preprocessing, metrics, svm,linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve,  explained_variance_score, precision_recall_curve,average_precision_score,accuracy_score, classification_report
#from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from scipy.stats import boxcox
from matplotlib import pyplot

import pickle
#from sklearn.externals import joblib
import joblib


# Load Image & Logo
#====================
st.image("Logo_AiX.jpg") # Change to MSpace Logo
#st.write("https://www.ailluminate.com")
#st.image("LogoAiX1.jpg") # Change to MSpace Logo

st.markdown("<h1 style='text-align: left; color: turquoise;'>Ailysium: BioForecast Platform</h1>", unsafe_allow_html=True)
#st.markdown("<h1 style='text-align: left; color: turquoise;'>Train AI BioForecast Model (Realtime)</h1>", unsafe_allow_html=True)

#st.markdown("<h1 style='text-align: left; color: turquoise;'>Opening-Economy & Society</h1>", unsafe_allow_html=True)



#df_forecast= pd.read_csv("2021-03-27-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 

df_forecast=pd.read_csv("recent-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 


#Load Data - The last/most recent Forecast and latest Data
#=====================

# The last two, most recent forecast

#df_forecast= pd.read_csv("2021-03-15-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
#Forcast_date="2021-03-15"
#Forecasted_dates=["3/20/2021", "3/27/2021", "4/03/2021", "4/10/2021" ]


#df_forecast= pd.read_csv("2021-03-22-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
#Forcast_date="2021-03-22"
#Forecasted_dates=["3/27/2021", "4/03/2021", "4/10/2021", "4/17/2021" ]

#==========================================


df_forecast_previous= pd.read_csv("previous-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
Forcast_date="2021-03-22"
Forecasted_dates=["3/27/2021", "4/03/2021", "4/10/2021", "4/17/2021" ]


df_forecast_recent=pd.read_csv("recent-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
Forcast_date="2021-03-29"
Forecasted_dates=["4/03/2021", "4/10/2021", "4/17/2021", "4/24/2021" ]





#================
#initialize the data
#=======================
#Models
#====================
#st.success("What Forecast Model Data to Load?")
forecast_model_Options= ['Reference Model',
        'Ensemble',
        'UGA-CEID',
        'Columbia',
        'ISU',
        'UVA',
        'LNQ',
        'Facebook',
        'JHU-APL',
        'UpstateSU',
        'JHU-IDD',
        'LANL',
        'Ensemble']

#st.success("What Date Forecast Data to Load?")
data_dates_options=['2021-01-04', '2021-01-11',   '2021-01-18',  
    '2021-01-25',  '2021-02-01',  '2021-02-08',  
    '2021-02-15',  '2021-02-22',  '2021-03-01',  
    '2021-03-08',  '2021-03-15',  '2021-03-22',
    '2021-03-27'] 

data_dates_options=['2021-03-29',
'2021-03-22', '2021-03-15',  '2021-03-08',
    '2021-03-01',   '2021-02-22',  '2021-02-15', 
    '2021-02-08',  '2021-02-01',   '2021-01-25',  
    '2021-01-18',  '2021-01-11',  '2021-01-04']  


data_dates_options=['2021-04-12']
    

load_ai_model_options=['Reference Model',
'AI Model 1',
'AI Model 2 (L)',
'AI Model 3 (Fast)',
'AI Model 4 (Fast) (L)',
'AI Model 5',
'AI Model 6',
'AI Model 7 (VERY Slow- Do Not Use, if You have too!)',
'AI Model 8',
'AI Model 9 (Slow)',
'AI Model 10',
'AI Model 11 (L)',
'AI Model 12',
'AI Model 13',
'AI Model 14 (L)',
'AI Model 15',
'AI Model 16 (L)',
'AI Model  (aggregator)']

train_ai_model_options=load_ai_model_options

#===========================

#Selectt Option Section
#============================

select_options=["AiX-ai-Forecast-Platform",
                "Load Forecast Data",  #Simply Check the Forecast Data 
                "Load AI Model",
                "Train AI Model",
                "AiX-Platform"]
                


select_options=["AiX-ai-Forecast-Platform"]              
your_option=select_options
st.sidebar.success("Please Select your Option" )
option_selectbox = st.sidebar.selectbox( "Select your Option:", your_option)  
select_Name=option_selectbox


#if option_selectbox=='Load Forecast Data' or option_selectbox!='Load Forecast Data':
#if select_Name=='Load Forecast Data' or select_Name!='Load Forecast Data':
if select_Name=='AiX-ai-Forecast-Platform' or select_Name!='AiX-ai-Forecast-Platform':

#Models
#====================
    #st.success("What Forecast Model Data to Load?") 
    your_option=forecast_model_Options
    st.sidebar.info("Please Select Forecast Model" )
    option_selectbox = st.sidebar.selectbox( "Select Forecast Model:", your_option)
    if option_selectbox =='Reference Model':
       option_selectbox='Reference Model'
       option_selectbox='Ensemble'
    forecast_model_Name=option_selectbox

#if option_selectbox=='Load Forecast Data' or option_selectbox!='Load Forecast Data':       
if select_Name=='Load Forecast Data' or select_Name!='Load Forecast Data':
    #st.success("What Date Forecast Data to Load?")

    your_option=data_dates_options
    st.sidebar.warning("Please Select Forecast Date" )
    option_selectbox = st.sidebar.selectbox( "Select Forecast Date:", your_option)
    #if option_selectbox=='2021-03-22':
    #   option_selectbox= '2021-03-15'
    data_dates_Name=option_selectbox
    
    
    
    if option_selectbox==data_dates_Name:
        your_option=["One(1) Week Ahead", "Two(2) Weeks Ahead", "Three(3) Weeks Ahead", "Four(4) Weeks Ahead"]
        st.sidebar.warning("Please Select Forecast Week" )
        option_selectbox = st.sidebar.selectbox( "Select Forecast Weeks Ahead:", your_option)  
        data_week_Name=option_selectbox


if data_week_Name !="One(1) Week Ahead":
    st.write("Two(2), Three(3), and Four(4) Weeks Ahead are being calculated offline currently and are not presented as realtime")

#if option_selectbox=='Load AI Model':
if select_Name=='Load AI Model':

    your_option=load_ai_model_options
    st.sidebar.error("Please Select AI Model  to load" )
    option_selectbox = st.sidebar.selectbox( "Select AI-Model to Load:", your_option)  
    ai_load_Name=option_selectbox

#if option_selectbox=='Train AI Model':
if select_Name=='Train AI Model':

    your_option=train_ai_model_options
    st.sidebar.success("Please Select AI Model to Train" )
    option_selectbox = st.sidebar.selectbox( "Select AI-Model to Train:", your_option)  
    ai_train_Name=option_selectbox


#load_data_csv=data_dates_Name+"-all-forecasted-cases-model-data.csv"

#st.write("Data to load: ", load_data_csv)



#Load Models and Sidebar Selection 
#===================================================================================# Load AI Models 

#if option_selectbox=='AiX Platform':
if select_Name=='AiX Platform':

    model2load=pd.read_csv('model2load.csv', engine='python', dtype=str) # dtype={"Index": int})


    model_index=model2load
    model_names_option=model_index.AI_Models.values

    st.sidebar.success("Please Select your AI Model!" )
    model_selectbox = st.sidebar.selectbox( "Select AI Model", model_names_option)  
    Model_Name=model_selectbox

    Index_model=model2load.Index[model2load.AI_Models==Model_Name].values[0]

    Index_model=int(Index_model)

    pkl_model_load=model2load.Pkl_Model[model2load.AI_Models==Model_Name].values[0]



    #Load Data and Model
    Pkl_Filename = pkl_model_load  #"Pickle_RForest.pkl"  
    #st.write(Pkl_Filename)
    # Load the Model back from file
    #****with open(Pkl_Filename, 'rb') as file:     # This line to load the file
    #***    Pickle_LoadModel = pickle.load(file)    # This line to load the file
    #   Pickle_RForest = pickle.load(file)
        #RForest=Pickle_RForest




load_data_csv=data_dates_Name+"-all-forecasted-cases-model-data.csv"

#st.write('Load CDC Model Data- Data to load:', '   ', load_data_csv)

load_data_csv="recent-all-forecasted-cases-model-data.csv"

#st.write('Load CDC Model Data- Data to load:', '   ', load_data_csv)

#Forecast Data is being loaded and alll sort of sidebars also created. 
#===================================================



#import pandas as pd



# Load Reference Model Forecast Ensemble - Only For Visualization Purpose
#=============================================================================
#df_forecast= pd.read_csv("2021-03-15-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
#df_forecast= pd.read_csv(load_data_csv,  engine='python', dtype={'fips': str}) 

df_forecast_ref=pd.DataFrame()
df_forecast_ref=pd.read_csv("previous-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
Forcast_date="2021-03-22"
Forecasted_dates=["3/27/2021", "4/03/2021", "4/10/2021", "4/17/2021" ]

df_forecast=pd.DataFrame()
df_forecast= df_forecast_ref.copy()
df=pd.DataFrame()
df=df_forecast.copy()

# Drop all the States. We are only interested in Counties
df_drop=df[df.location_name!=df.State]
#df_drop1 = df.query("location_name != State")
#df_drop.fips= df_drop.fips.astype(str)
df_forecast=df_drop.copy()

#df_drop.fips= df_drop.fips.astype(str)
#df_forecast_Ensemble=df_forecast[df_forecast.model=="Ensemble"]
 
#forecast_model_Name="Ensemble"

df_forecast_Ensemble=pd.DataFrame()
df_forecast_Ensemble=df_forecast[df_forecast.model=="Ensemble"]
df_forecast_Ensemble=df_forecast_Ensemble[df_forecast_Ensemble.target=="1 wk ahead inc case"]


df_forecast_Ensemble_ref=pd.DataFrame()
df_forecast_Ensemble_ref=df_forecast_Ensemble.copy()






# Load Previous Forecast
#=========================
#df_forecast= pd.read_csv("2021-03-15-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
#df_forecast= pd.read_csv(load_data_csv,  engine='python', dtype={'fips': str}) 

df_forecast_previous=pd.DataFrame()
df_forecast_previous=pd.read_csv("previous-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
Forcast_date="2021-03-22"
Forecasted_dates=["3/27/2021", "4/03/2021", "4/10/2021", "4/17/2021" ]

df_forecast=pd.DataFrame()
df_forecast= df_forecast_previous.copy()
df=pd.DataFrame()
df=df_forecast.copy()

# Drop all the States. We are only interested in Counties
df_drop=df[df.location_name!=df.State]
#df_drop1 = df.query("location_name != State")
#df_drop.fips= df_drop.fips.astype(str)
df_forecast=df_drop.copy()

#df_drop.fips= df_drop.fips.astype(str)
#df_forecast_Ensemble=df_forecast[df_forecast.model=="Ensemble"]
 

df_forecast_Ensemble=pd.DataFrame()
df_forecast_Ensemble=df_forecast[df_forecast.model==forecast_model_Name]
df_forecast_Ensemble=df_forecast_Ensemble[df_forecast_Ensemble.target=="1 wk ahead inc case"]


df_forecast_Ensemble_previous=pd.DataFrame()
df_forecast_Ensemble_previous=df_forecast_Ensemble.copy()




#Load Most Recent Forecast
#====================

#df_forecast= pd.read_csv(load_data_csv,  engine='python', dtype={'fips': str}) 

df_forecast_recent=pd.DataFrame()
df_forecast_recent=pd.read_csv("recent-all-forecasted-cases-model-data.csv",  engine='python', dtype={'fips': str}) 
Forcast_date="2021-03-29"
Forecasted_dates=["4/03/2021", "4/10/2021", "4/17/2021", "4/24/2021" ]

df_forecast=pd.DataFrame()
df_forecast= df_forecast_recent.copy()
df=pd.DataFrame()
df=df_forecast.copy()

# Drop all the States. We are only interested in Counties
df_drop=df[df.location_name!=df.State]
#df_drop1 = df.query("location_name != State")
#df_drop.fips= df_drop.fips.astype(str)


#df_drop.fips= df_drop.fips.astype(str)
#df_forecast_Ensemble=df_forecast[df_forecast.model=="Ensemble"]
 


df_forecast_Ensemble=pd.DataFrame()
df_forecast_Ensemble=df_forecast[df_forecast.model==forecast_model_Name]
df_forecast_Ensemble=df_forecast_Ensemble[df_forecast_Ensemble.target=="1 wk ahead inc case"]


df_forecast_Ensemble_recent=pd.DataFrame()
df_forecast_Ensemble_recent=df_forecast_Ensemble.copy()





#Load Actual Cases
#==========================

df_actual_cases=pd.DataFrame()
df_actual_cases=pd.read_csv("covid_confirmed_usafacts_forecast.csv",  engine='python', dtype={'fips': str}) 




#======================Visulaization of data =======================
# ======================Compare the Forecast with actula data ================"



df_ref_temp=pd.DataFrame(np.array(df_forecast_Ensemble_ref.iloc[:,[6,7]].values), columns=["fips", "Forecast_Reference"])  # 6,7: fips and point 


df_model_temp=pd.DataFrame(np.array(df_forecast_Ensemble_previous.iloc[:,[6,7]].values), columns=["fips", "Forecast_Model"]) # 6,7: fips and point


df_actual_temp=pd.DataFrame(np.array(df_actual_cases.iloc[:,[0,-2]].values), columns=["fips", "Actual_Target"])  # 0, -2: fips and most recent actual-target

df_actual_temp=pd.DataFrame(np.array(df_actual_cases.iloc[:,[0,-7,-6,-5,-4,-3, -2]].values), 
    columns=["fips", "TimeN5",	"TimeN4",	"TimeN3",	"TimeN2",	"TimeN1", "Actual_Target"])  # 0, -2: fips and most recent actual-target


#st.write("Last 6 Total Weekly Cases, ", df_actual_temp.head(20))






data_merge= pd.DataFrame() #df_ref_temp.copy()
data_merge= pd.merge(df_ref_temp, df_model_temp, on="fips")
data_merge_left=data_merge.copy()
data_merge= pd.merge(data_merge_left, df_actual_temp, on="fips")


#st.write("df_actual_temp:, ", data_merge.head())

#st.error("Stop for checking how many is loaded")




data_merge.iloc[:,1:] = data_merge.iloc[:,1:].astype(float)



#st.write("Data Merged:  ", data_merge.head())


#data_merge = data_merge.iloc[:,[1,2,3]].astype(float)

df_forecast_target=data_merge.copy()

#df_forecast_target_Scaled = df_forecast_target_Scaled.astype(float)


len_data=len(df_forecast_target)

df_population= pd.read_csv("covid_county_population_usafacts.csv",  engine='python', dtype={'fips': str, 'fips_1': str})

df_forecast_target_Scaled = df_forecast_target.copy()



i=0
while i <len_data: 
  
      fips=df_forecast_target['fips'].iloc[0]
      population=df_population.population[df_population.fips==fips].values[0]
      df_forecast_target_Scaled.iloc[i,1:]=df_forecast_target.iloc[i,1:]/population*1000
      i=i+1




df_forecast_target_Scaled.iloc[:,1:] = df_forecast_target_Scaled.iloc[:,1:].astype(float)

#st.write("df_forecast_target_Scaled", df_forecast_target_Scaled.head())


data_viz=df_forecast_target_Scaled.copy()



#Delete All The Data Frames that we do not need!
#=======================Delete all the DataFrame we do not need ==================
df_forecast_target_Scaled=pd.DataFrame()
data_merge=pd.DataFrame()
df_forecast_target=pd.DataFrame()

df_forecast_Ensemble_previous=pd.DataFrame()
df_forecast_Ensemble_recent=pd.DataFrame()
df_forecast_Ensemble_ref=pd.DataFrame()
df_forecast=pd.DataFrame()
df_ref_temp=pd.DataFrame()
df_model_temp=pd.DataFrame()
df_actual_temp=pd.DataFrame()
df_drop=pd.DataFrame()





#data_viz.to_csv("data_viz.csv", index=False)

data_viz= data_viz.drop(data_viz.columns[[0]], axis=1)

data_viz= data_viz*100

data_viz= data_viz.astype(float)

#st.write("Data viz: head ", data_viz.head())
#st.write("Data viz: Stat ", data_viz.describe())


data_viz.drop( data_viz[ data_viz.Forecast_Reference >4500 ].index , inplace=True)

data_viz.drop( data_viz[ data_viz.Forecast_Model >4500 ].index , inplace=True)

data_viz.drop( data_viz[ data_viz.Actual_Target >5000 ].index , inplace=True)


data_viz.drop( data_viz[ data_viz.TimeN1>5000 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN2>5000 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN3>5000 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN4>5000 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN5>5000 ].index , inplace=True)





#st.write("Data viz: Stat 2- after cut off of 4500-5000 ", data_viz.describe())


#st.success("Stop")


#data_viz= data_viz*100

#data_viz["Forecast_Reference"]=data_viz["Forecast_Reference"].apply(np.ceil)
data_viz.drop( data_viz[ data_viz.Forecast_Reference <1 ].index , inplace=True)

#data_viz["Forecast_Model"]=data_viz["Forecast_Model"].apply(np.ceil)
data_viz.drop( data_viz[ data_viz.Forecast_Model <1 ].index , inplace=True)

#data_viz["Actual_Target"]=data_viz["Actual_Target"].apply(np.ceil)
data_viz.drop( data_viz[ data_viz.Actual_Target <1 ].index , inplace=True)


data_viz.drop( data_viz[ data_viz.TimeN1<1 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN2<1 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN3<1 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN4<1 ].index , inplace=True)
data_viz.drop( data_viz[ data_viz.TimeN5<1 ].index , inplace=True)






#data_viz= np.around(data_viz)
#data_viz=data_viz[data_viz.Actual_Target>=0]
#data_viz= data_viz*100 #np.around(data_viz+)  

#data_viz=data_viz[data_viz.Actual_Target<5000]
#data_viz=data_viz[data_viz.Forecast_Reference<4200]
#data_viz=data_viz[data_viz.Forecast_Model<4200]
#data_viz_temp=data_viz[data_viz<5000]


if data_viz.empty:
    st.error("No Data matches our criteria both for AI Model and Visualization!")
    st.warning("Please select another option!")
    st.stop("The Program stopped here!")

#data_viz.drop( data_viz[ data_viz >5000 ].index , inplace=True)

#st.write("describe data -2")
#st.write(data_viz.describe())


#================= Visualization
#sns.jointplot(data=data_viz, x="target", y="Ensemble")

#sns.pairplot(data=data_viz, hue='color')
#data_viz=pd.read_csv("data_viz.csv",  engine='python')

i=0.2
data_viz=(data_viz**i-1)/i
#data_viz=np.log(data_viz)

#st.write("Data viz: Stat3333333333333 ", data_viz.describe())

huecolor=data_viz.Actual_Target.values
huecolor=huecolor.astype(int)

data_viz["huecolor"]=huecolor.astype(int)

#data_viz=data_viz[data_viz>0]

#st.write("describe data -2")
#st.write(data_viz.describe())

huecolor=data_viz.Actual_Target.values.astype(int)

data_viz["huecolor"]=huecolor.astype(int)


#st.title("Hello")
#fig = sns.pairplot(penguins, hue="species")
#st.pyplot(fig)

data_vis=data_viz.copy()



st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'>Forecast: Reference vs Selected Model </h1>", unsafe_allow_html=True)


#fig=sns.pairplot(data_viz, hue="huecolor", diag_kind="hist")
#st.pyplot(fig)

data_vis= data_vis.drop(data_vis.columns[[2,3,4,5,6]], axis=1)

fig=sns.pairplot(data_vis, hue="huecolor", diag_kind="hist")
st.pyplot(fig)

#data_vis=pd.DataFrame()

#import numpy as np
#import pandas as pd
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
#import matplotlib.pyplot as plt

mod = smf.quantreg('Forecast_Model ~ Actual_Target', data_viz)
res = mod.fit(q=.5)


#st.write(res.summary())
#LRresult = (res.summary2().tables[0])
#st.write(LRresult)
#LRresult = (res.summary2().tables[1])
#st.write(LRresult)



#import statsmodels.api as sm

#model = sm.OLS(y,x)
#res = model.fit()
results_summary = res.summary()


results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)


#print("Stop the program here")
#st.stop()

#import statsmodels.api as sm
#from scipy.stats import boxcox
#from matplotlib import pyplot
#hist=pyplot.hist(data_viz['Forecast_Model'],100)

#fig = plt.figure()
#plt.hist(data_viz['Forecast_Model'],100)
#st.plotly_chart(fig)




#====================Stat plot ================================

#mod = smf.quantreg('AiX_AI_Model3 ~ target', covid19_forecast)
#========================================

quantiles = np.arange(.025, .96, .1)
quantiles = np.arange(.05, 0.975, .125)
quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]

def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['Actual_Target']] + \
            res.conf_int().loc['Actual_Target'].tolist()

models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

ols = smf.ols('Forecast_Model ~ Actual_Target', data_viz).fit()
ols_ci = ols.conf_int().loc['Actual_Target'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['Actual_Target'],
           lb = ols_ci[0],
           ub = ols_ci[1])

#st.write("Models: ", models)
#st.write("OLS: ", ols)




x = np.arange(data_viz.Actual_Target.min(), data_viz.Actual_Target.max(), 20)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(18, 10))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')

y = get_y(ols['a'], ols['b'])


ax.plot(x, y, color='red', label='OLS-Map')
ax.scatter(data_viz.Actual_Target, data_viz.Forecast_Model, color='blue',   alpha=.2)
ax.set_xlim((-1, 20))
ax.set_ylim((-1, 20))
legend = ax.legend()
ax.set_xlabel('Covid19 Actual Cases', fontsize=16)
ax.set_ylabel('Covid19 Forecast/Predictions Cases (Forecast_Model)', fontsize=16);


st.pyplot(fig)






#st.write(data_viz.head())


#AI Section starts from here
#===========================AiX-AI Section=================================

#data = data_viz.iloc[:,[0]]   # 0,1
#target=data_viz.iloc[:,2].values


data = data_viz.iloc[:,[0,2,3,4,5,6]]
#target=data_viz.iloc[:,7].values

target=data_viz.Actual_Target.values


data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10) #stratify=np.ravel(target))
train =data_train
test =data_test
train_target=target_train
test_target=target_test

X_train, X_val, y_train, y_val = train_test_split(train, train_target, test_size=0.3, shuffle=True)


data=np.round(data*100,0)
target=np.round(target*100,0)
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10) #stratify=np.ravel(target))
train =data_train
test =data_test
train_target=target_train
test_target=target_test

X_train, X_val, y_train, y_val = train_test_split(train, train_target, test_size=0.3, shuffle=True)


#Save the Model RF

#==========Random Forest===========================

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
rng = np.random.RandomState(1)

#D3Forest = DecisionTreeClassifier(max_depth=10, random_state=1)
#D3Forest = DecisionTreeRegressor(random_state=0)
#D3Forest.fit(data_train, np.ravel(target_train))

#RForest = RandomForestClassifier(max_depth=10, random_state=1)
#RForest = RandomForestClassifier(n_estimators=50)  # 128 Optimal


#RForest = RandomForestRegressor(max_depth=30, random_state=0)
#RForest = RandomForestRegressor(n_estimators=300, random_state=rng)
RForest = RandomForestRegressor(random_state=rng)


RForest.fit(data_train, np.ravel(target_train))

pred = RForest.predict(data_test)
pred_rf= RForest.predict(data)
pred=np.round(pred,0)
pred_rf=np.round(pred_rf,0)


rf_acc=1-accuracy_score(np.ravel(target_test), pred)


#Save the Model RF
#=====================
#import pickle
#from sklearn.externals import joblib
#import joblib

# Save the Model to file in the current working directory
Pkl_Filename = "Pickle_ML_RForest.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(RForest, file)

#pickle.dump(RForest, open("test_RForest.pkl", 'wb'))
#joblib.dump(RForest, "test_joblip.pkl", 9)

joblip_Filename = "joblip_ML_RForest.pkl" 
with open(joblip_Filename, 'wb') as file:  
    joblib.dump(RForest, file,9)

#with open(joblip_Filename, 'rb') as file:  
#    joblip_RForest=joblib.load(file)

# Load the Model back from file
#Pkl_Filename = "Pickle_ML_RForest.pkl" 
#with open(Pkl_Filename, 'rb') as file:  
#    Pickle_RForest = pickle.load(file)
#RForest1=Pickle_RForest
#pred1 = RForest1.predict(data_test)
#pred1=np.round(pred,0)
#rf_acc1=1-accuracy_score(np.ravel(target_test), pred1)

#with open(joblip_Filename, 'rb') as file:  
#    joblip_RForest=joblib.load(file)

#RForest2=joblip_RForest

#pred2 = RForest2.predict(data_test)
#pred2=np.round(pred,0)
#rf_acc2=1-accuracy_score(np.ravel(target_test), pred2)

#st.write(rf_acc, rf_acc1)
#st.write("RandomForest accuracy score : ",accuracy_score(np.ravel(target_test), pred))
#st.write("AiX-AI Model 1 accuracy score : ",accuracy_score(np.ravel(target_test), pred))




#===========================Ada-DRForest Mix ===========================================

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)
# Fit regression model
#regr_1 = DecisionTreeRegressor(max_depth=10)

#AdaRForest = AdaBoostRegressor(RandomForestRegressor(max_depth=5),
#                          n_estimators=200, random_state=rng)

AdaRForest = AdaBoostRegressor(RandomForestRegressor(max_depth=5), random_state=rng)

#regr_1.fit(data_train, target_train)
AdaRForest.fit(data_train, target_train)

# Predict

pred = AdaRForest.predict(data_test)
pred_adaRF= AdaRForest.predict(data)

#===============
#def f(x):
#    return np.int(x)
#f2 = np.vectorize(f)
#pred=f2(pred)
#pred_rf_reg=f2(pred_rf_reg)
#================

pred=np.round(pred,0)
pred_adaRF=np.round(pred_adaRF,0)

adaRF_acc=1-accuracy_score(np.ravel(target_test), pred)

st.write("AiX-AI Model Ada-Random Forest (RF) accuracy score : ", 1-adaRF_acc) 



# Save the Model to file in the current working directory
Pkl_Filename = "Pickle_ML_AdaRForest.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(AdaRForest, file)
    
joblip_Filename = "joblip_ML_AdaRForest.pkl" 
with open(joblip_Filename, 'wb') as file:  
    joblib.dump(AdaRForest, file,9)    

# Save the Model to file in the current working directory
#Pkl_Filename = "Pickle_ML_RForest_4ada.pkl"  #
#with open(Pkl_Filename, 'wb') as file:  
#    pickle.dump(regr_1, file)





#===========================Ada-D3Forest Mix ===========================================

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)
# Fit regression model
#regr_1 = DecisionTreeRegressor(max_depth=10)

#AdaD3Forest = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
#                          n_estimators=200, random_state=rng)

AdaD3Forest = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), random_state=rng)
                          
#regr_1.fit(data_train, target_train)
AdaD3Forest.fit(data_train, target_train)

# Predict

pred = AdaD3Forest.predict(data_test)
pred_adaD3= AdaD3Forest.predict(data)

#===============
#def f(x):
#    return np.int(x)
#f2 = np.vectorize(f)
#pred=f2(pred)
#pred_rf_reg=f2(pred_rf_reg)
#================

pred=np.round(pred,0)
pred_adaD3=np.round(pred_adaD3,0)

adaD3_acc=1-accuracy_score(np.ravel(target_test), pred)

st.write("AiX-AI Model Ada-Decision Tree(D3) accuracy score : ", 1-adaD3_acc) 



# Save the Model to file in the current working directory
Pkl_Filename = "Pickle_ML_adaD3Forest.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(AdaD3Forest, file)
    
joblip_Filename = "joblip_ML_adaD3Forest.pkl" 
with open(joblip_Filename, 'wb') as file:  
    joblib.dump(AdaD3Forest, file,9)    

# Save the Model to file in the current working directory
#Pkl_Filename = "Pickle_ML_RForest_4ada.pkl"  #
#with open(Pkl_Filename, 'wb') as file:  
#    pickle.dump(regr_1, file)




#============================== Decision Tree- D3 =================================

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
rng = np.random.RandomState(1)

#D3Forest = DecisionTreeClassifier(max_depth=10, random_state=1)

#D3Forest = DecisionTreeRegressor(max_depth=10, random_state=rng)
D3Forest = DecisionTreeRegressor(random_state=rng)
D3Forest.fit(data_train, np.ravel(target_train))

pred = D3Forest.predict(data_test)
pred_d3= D3Forest.predict(data)
pred=np.round(pred,0)
pred_d3=np.round(pred_d3,0)

d3_acc=1-accuracy_score(np.ravel(target_test), pred)

st.write("AiX-AI Model Decision Tree(D3) accuracy score : ", 1-d3_acc) 


#Save the Model RF
#=====================
#import pickle
#from sklearn.externals import joblib
#import joblib

# Save the Model to file in the current working directory
Pkl_Filename = "Pickle_ML_D3Forest.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(D3Forest, file)

#pickle.dump(RForest, open("test_RForest.pkl", 'wb'))
#joblib.dump(RForest, "test_joblip.pkl", 9)

joblip_Filename = "joblip_ML_D3Forest.pkl" 
with open(joblip_Filename, 'wb') as file:  
    joblib.dump(D3Forest, file,9)



#=====================KNN ===========================

#A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
#import necessary modules
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=1)
#Train the algorithm
neigh.fit(data_train, np.ravel(target_train))
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
#st.write("KNeighbors accuracy score : ",accuracy_score(np.ravel(target_test), pred))
#st.write("AiX-AI Model 2 accuracy score : ",accuracy_score(np.ravel(target_test), pred))
pred_knn=neigh.predict(data)

pred=np.round(pred,0)
pred_knn=np.round(pred_knn,0)

knn_acc=1-accuracy_score(np.ravel(target_test), pred)

#Save the Model KNN
#=====================
#import pickle
# Save the Model to file in the current working directory
Pkl_Filename = "Pickle_ML_KNN.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(neigh, file)


# Load the Model back from file
#Pkl_Filename = "Pickle_ML_KNN.pkl"  
#with open(Pkl_Filename, 'rb') as file:  
#    Pickle_RForest = pickle.load(file)
#neigh1=Pickle_RForest
#pred1 = neigh1.predict(data_test)
#pred1=np.round(pred,0)
#knn_acc1=1-accuracy_score(np.ravel(target_test), pred1)

#st.write(knn_acc, knn_acc1)



#=============MLP =============================

#from sklearn.neural_network import MLPRegressor
#from sklearn.datasets import make_regression
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler

#regr = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000).fit(data_train, target_train)
#from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(data_train)
X_testscaled=sc_X.transform(data_test)
X_data=sc_X.transform(data)
#regr = MLPRegressor(hidden_layer_sizes=(25,25),activation="relu" ,random_state=1, max_iter=2000).fit(data_train, target_train)

regr = MLPRegressor(random_state=1, max_iter=5000).fit(data_train, target_train)

pred = regr.predict(data_test)
#st.write ("MLP accuracy score : ",accuracy_score(np.ravel(target_test), pred))
#st.write ("AiX-AI Model 3 accuracy score : ",accuracy_score(np.ravel(target_test), pred))
pred_MLP=regr.predict(data)

pred=np.round(pred,0)
pred_MLP=np.round(pred_MLP,0)


mlp_acc=1-accuracy_score(np.ravel(target_test), pred)

#regr1 = MLPRegressor(hidden_layer_sizes=(10,5,2),activation="relu" ,random_state=1, max_iter=2000).fit(X_trainscaled, target_train)
#pred = regr1.predict(X_testscaled)
#pred=np.round(pred,0)

#pred_MLP=regr1.predict(X_data)
#mlp_acc_scale=1-accuracy_score(np.ravel(target_test), pred)


#Save the Model MLP
#=====================
#import pickle
# Save the Model to file in the current working directory
Pkl_Filename = "Pickle_ML_MLP.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(regr, file)


# Load the Model back from file
#Pkl_Filename = "Pickle_ML_MLP.pkl"  
#with open(Pkl_Filename, 'rb') as file:  
#    Pickle_RForest = pickle.load(file)

#regr1=Pickle_RForest
#pred1 = regr1.predict(data_test)
#pred1=np.round(pred,0)
#mlp_acc1=1-accuracy_score(np.ravel(target_test), pred1)

#st.write(mlp_acc, mlp_acc1, mlp_acc_scale)


#==============AI Aggregator =====================

#df_MLearning=pd.DataFrame({"RF":pred_rf, "KNN":pred_knn, "MLP":pred_MLP})

#df_MLearning=pd.DataFrame({"AiX_AI_Model1":pred_rf, "AiX_AI_Model2":pred_knn, "AiX_AI_Model3":pred_MLP})

df_MLearning=pd.DataFrame({"AiX_AI_Model1":pred_rf, "AiX_AI_Model2":pred_knn, "AiX_AI_Model3":pred_MLP,   
                             "AiX_AI_Model1D3":pred_d3, "AiX_AI_Model1D3Ada":pred_adaD3, "AiX_AI_Model1RFAda":pred_adaRF })


st.write(df_MLearning.head(20))

#df_MLearning["ML_Median"]=df_MLearning.median(axis=1)
#df_MLearning["ML_Mean"]=df_MLearning.mean(axis=1)

#df_MLearning["AiX_AI_Model4"]=df_MLearning.median(axis=1)
#df_MLearning["AiX_AI_Model5"]=df_MLearning.iloc[:,[0,1,2]].mean(axis=1)


df_MLearning["AiX_AI_Model4"]=df_MLearning.iloc[:,[0,1,2, 3]].median(axis=1)
#df_MLearning["AiX_AI_Model4"]=df_MLearning.median(axis=1)
df_MLearning["AiX_AI_Model5"]=df_MLearning.iloc[:,[0,1,2,3]].mean(axis=1)




#df_smart= (df_MLearning["AiX_AI_Model1"]*rf_acc + df_MLearning["AiX_AI_Model2"]*knn_acc+ 
#                       df_MLearning["AiX_AI_Model3"]*mlp_acc+df_MLearning["AiX_AI_Model4"]*4+   # Median 75% - Mean 25% 
#                       df_MLearning["AiX_AI_Model5"]*2)/(rf_acc+knn_acc+mlp_acc+6)



df_smart= (df_MLearning["AiX_AI_Model1"]*rf_acc + df_MLearning["AiX_AI_Model2"]*knn_acc+ df_MLearning["AiX_AI_Model3"]*mlp_acc +
                       df_MLearning["AiX_AI_Model1D3"]*d3_acc+df_MLearning["AiX_AI_Model4"]*4+   # Median 75% - Mean 25% 
                       df_MLearning["AiX_AI_Model5"]*2)/(rf_acc+knn_acc+mlp_acc+d3_acc+6)

df_MLearning["AiX_AI_Smart_Committee_Machine"]=df_smart #   df_MLearning.mean(axis=1)


df_MLearning["target"]=target

df_MLearning1=df_MLearning.copy()/100
df_MLearning1=df_MLearning1.astype(int)


#st.write(df_MLearning1.head())


st.write("df_MLearning1:  ", df_MLearning1.head(20))
df_MLearning1=pd.DataFrame()


# No Need for Deployed App
#======================================================

#st.write(" ")
#st.markdown("<h1 style='text-align: left; color: red;'> AiX - AI Smart Committee Machine</h1>", unsafe_allow_html=True)

#=============ML Plots================

data_ML=df_MLearning.copy()/100

df_MLearning["huecolor"]= data_viz["huecolor"].values  #(huecolor).astype(int)
data_ML["huecolor"]=  data_viz["huecolor"].values #  (huecolor).astype(int)





df_MLearning=pd.DataFrame()

#fig=sns.pairplot(data_ML, hue="huecolor", diag_kind="hist")

#st.pyplot(fig)





#======================ML Output ======================

covid19_forecast=data_ML.iloc[:,]




#median_acc=np.median([rf_acc, knn_acc, mlp_acc])
#mean_acc=np.mean([rf_acc, knn_acc, mlp_acc])
#smart_acc=1-(rf_acc+knn_acc+ mlp_acc+ 4*median_acc+ 2*mean_acc)/(rf_acc+knn_acc+mlp_acc+6)

median_acc=np.median([mlp_acc, knn_acc, mlp_acc, d3_acc])
mean_acc=np.mean([mlp_acc, knn_acc, mlp_acc, d3_acc])
smart_acc=1-(mlp_acc+knn_acc+ mlp_acc+ d3_acc+ 4*median_acc+ 2*mean_acc)/(mlp_acc+knn_acc + mlp_acc+d3_acc+6)


#===========Smart Committee Machine #======================== 

st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Smart Committee Machine</h1>", unsafe_allow_html=True)
st.write("AiX-AI Smart Committee Machine< accuracy score : ", smart_acc) 

mod = smf.quantreg('AiX_AI_Smart_Committee_Machine ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)




#============== Model 1 ============================

st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model1</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 1 accuracy score : ", 1-rf_acc) 

mod = smf.quantreg('AiX_AI_Model1 ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)


#============== Model 1-2 D3============================

st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model1 - D3</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 1-2 D3 accuracy score : ", 1-d3_acc) 

mod = smf.quantreg('AiX_AI_Model1D3 ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)


#============== Model 1-3 ============================

st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model1 - Ada-D3</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 1-3 ada-D3 accuracy score : ", 1-adaD3_acc) 

mod = smf.quantreg('AiX_AI_Model1D3Ada ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)

#============== Model 1-4 ============================

st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model1 - Ada-RF</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 1-4 adaRF accuracy score : ", 1-adaRF_acc) 

mod = smf.quantreg('AiX_AI_Model1RFAda ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)




#============== Model 2 ============================


st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model2</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 2 accuracy score : ", 1-knn_acc)

mod = smf.quantreg('AiX_AI_Model2 ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)




#============== Model 3 ============================


st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model3</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 3 accuracy score : ", 1-mlp_acc) 

mod = smf.quantreg('AiX_AI_Model3 ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)

#=================


#============== Model 4 ============================


st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model4</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 4 accuracy score : ", 1-np.median([rf_acc, knn_acc, mlp_acc, d3_acc]))

mod = smf.quantreg('AiX_AI_Model4 ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)




#============== Model 5 ============================


st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Model5</h1>", unsafe_allow_html=True)
st.write("AiX-AI Model 5 accuracy score : ", 1-np.mean([rf_acc, knn_acc, mlp_acc, d3_acc]))

mod = smf.quantreg('AiX_AI_Model5 ~ target', covid19_forecast)
res = mod.fit(q=.5)

results_summary = res.summary()

results_as_html = results_summary.tables[0].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
st.write(df_res)

# Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
results_as_html = results_summary.tables[1].as_html()
df_res=pd.read_html(results_as_html, header=0, index_col=0)[0]
#st.write(df_res)






#================================== Final Results ===========================================

st.write(" ")
st.markdown("<h1 style='text-align: left; color: turquoise;'> Forecast: AiX vs Reference Models</h1>", unsafe_allow_html=True)

data_ML_Ensemble=pd.DataFrame()
data_ML_Ensemble["Target-Forecast"]=data_ML["target"]


data_ML_Ensemble["Forecast_Reference"]=data_viz['Forecast_Reference'].values
#data_ML_Ensemble["Forecast_Model"]=data_viz['Forecast_Model'].values
data_ML_Ensemble["AiX_Smart_Committee_Machine"]=data_ML['AiX_AI_Smart_Committee_Machine'].values
data_ML_Ensemble["huecolor"]=data_ML["huecolor"].values
fig=sns.pairplot(data_ML_Ensemble, hue="huecolor", diag_kind="hist")

st.pyplot(fig)

data_ML=pd.DataFrame()
data_ML_Ensemble=pd.DataFrame()

st.markdown("<h1 style='text-align: left; color: turquoise;'> AiX - AI Smart Committee Machine </h1>", unsafe_allow_html=True)

#====================Stat plot ================================

#mod = smf.quantreg('AiX_AI_Model3 ~ target', covid19_forecast)
#========================================

quantiles = np.arange(.025, .96, .1)
quantiles = np.arange(.05, 0.975, .125)
quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]

def fit_model(q):
    res = mod.fit(q=q)
    return [q, res.params['Intercept'], res.params['target']] + \
            res.conf_int().loc['target'].tolist()

models = [fit_model(x) for x in quantiles]
models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])

ols = smf.ols('AiX_AI_Smart_Committee_Machine  ~ target', covid19_forecast).fit()
ols_ci = ols.conf_int().loc['target'].tolist()
ols = dict(a = ols.params['Intercept'],
           b = ols.params['target'],
           lb = ols_ci[0],
           ub = ols_ci[1])

#st.write("Models: ", models)
#st.write("OLS: ", ols)




x = np.arange(covid19_forecast.target.min(), covid19_forecast.target.max(), 20)
get_y = lambda a, b: a + b * x

fig, ax = plt.subplots(figsize=(18, 10))

for i in range(models.shape[0]):
    y = get_y(models.a[i], models.b[i])
    ax.plot(x, y, linestyle='dotted', color='grey')

y = get_y(ols['a'], ols['b'])


ax.plot(x, y, color='red', label='OLS-Map')
ax.scatter(covid19_forecast.target, covid19_forecast.AiX_AI_Smart_Committee_Machine, color='blue',   alpha=.2)
ax.set_xlim((-1, 20))
ax.set_ylim((-1, 20))
legend = ax.legend()
ax.set_xlabel('Covid19 Actual Cases', fontsize=16)
ax.set_ylabel('Covid19 Forecast/Predictions Cases (AiX-AI Model)', fontsize=16);


st.pyplot(fig)



#============Last plot ===========================


st.write('Target: Actual Cases(Scaled) vs. Predicted/Forecast Cases (Scaled); One Week Ahead')

#fig = plt.figure()

fig, ax = plt.subplots(figsize=(18, 10))

#sns.lmplot(data=covid19_forecast, x="target", y="AiX_AI_Smart_Committee_Machine")  #, hue="target")
sns.scatterplot(data=covid19_forecast,x="target", y="AiX_AI_Smart_Committee_Machine", hue="target", size="target", sizes=(10, 100))   #, legend="False") # palette="deep")
st.pyplot(fig)


#title('Height vs Weight for Legendary and Non-Legendary Pokemons')
#fig2 = plt.figure()
#sns.scatterplot(data=df , x = 'weight_kg' , y = 'height_m' , hue='is_legendary')
#st.pyplot(fig2)


#plt.hist(data_viz['Forecast_Model'],100)
#st.pyplot()

#st.write(hist)








#===============================


y_actual=data_viz['Actual_Target']
fig, ax = plt.subplots(figsize=(18,9))
#ax.plot(y_actual, data_viz['Ensemble'], 'r.', label="Ensemble")
#ax.scatter(y_actual, data_viz['Ensemble'], s=area, alpha=0.5, label="Distance-Error") # c=colors, cmap='jet'

#plt.plot(covid19_forecast.target, covid19_forecast.AiX_AI_Smart_Committee_Machine,'ro', label="AiX - Smart Committee Machine")
#plt.plot(target/100,pred_knn/100,'bo', label="KNN")
#plt.plot(target/100,pred_MLP/100,'go', label="MLP")
#ax.plot(y_actual, data_viz['Ensemble'], 'mo', label="Ensemble")
#ax.plot(target/100, target/100, 'r--.', label="True")
ax.legend(loc='best');

plt.xlabel('Covid19 Actual Cases')
plt.ylabel('Covid19 Forecast/Predictions Cases')
plt.title('Bio-Forecasting')



x =  covid19_forecast.target #target_scaled
y = covid19_forecast.AiX_AI_Smart_Committee_Machine  #predictions_Scaled

N=len(x)
#colors = (.01*(x-y))**2
#area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

r = np.sqrt(abs(x*x-y*y))
area = np.pi*(1 * r)**2 # 0 to xx point radiuses
colors = np.sqrt(area)

#ax.plot(y_actual, data_viz['Ensemble'], 'r.', label="Ensemble")
ax.scatter(data_viz.Actual_Target, data_viz.Forecast_Model, s=area, alpha=0.5, label="Distance-Error") # c=colors, cmap='jet'

ax.plot(data_viz.Actual_Target, data_viz.Forecast_Model, 'go', label="Forecast Model")
ax.plot(data_viz.Actual_Target, data_viz.Forecast_Reference, 'yo', label="Reference Model")
plt.plot(covid19_forecast.target, covid19_forecast.AiX_AI_Smart_Committee_Machine,'ro', label="AiX - Smart Committee Machine")

ax.plot(y_actual, y_actual, 'r--.', label="True")
ax.legend(loc='best');

plt.xlabel('Covid19 Actual Cases')
plt.ylabel('Covid19 Forecast/Predictions Cases')
plt.title('Bio-Forecasting')


st.pyplot(fig)





st.write(" ")

st.image("ComparingForecast.jpg")



#st.info("Stop the Program Here")
#st.stop()


#============End of program============================================
#======================================================================

#================================================================================

st.markdown("<h1 style='text-align: left; color: turquoise;'>AiX - References :</h1>", unsafe_allow_html=True) 

st.write(" AiX- Covid19 Risk Analytics")
st.write("https://www.ailluminatex.org")
st.write("https://www.ailysium.com")
st.write("http://screening.ailysium.com")
st.write("http://triage.ailysium.com")




#st.image("Screening.jpg")
st.video("https://www.youtube.com/watch?v=JaAJ0rArXoc")
st.video("https://www.youtube.com/watch?v=gral3Wv45Go")

#st.video("https://www.youtube.com/channel/UCkfa2xCbUhHYXEm1qRkXXLA")

st.video("https://youtu.be/yLonTnRciXo")




##########################################################

#Update Dates

st.write("===============================================================")

last_update_dates=pd.read_csv("Lastupdate.csv",  engine='python', dtype= str) #, dtype={"countyFIPS1": str}, dtype={"stateFIPS": str} )
Initial_Model_Guidlines=last_update_dates.iloc[0,0]
AI_Model=last_update_dates.iloc[0,1]
Hybrid_AI_Guidlines=last_update_dates.iloc[0,2]
Next_Update=last_update_dates.iloc[0,3]

#=============================Legal and Privacy Notice ===========
# Side Bar Legal Notice


st.sidebar.write(" ")
st.sidebar.write("==================")
st.sidebar.markdown("<h1 style='text-align: left; color: turquoise;'>Privacy Policy&Legal:</h1>", unsafe_allow_html=True)
st.sidebar.write("Last Data updates:")
st.sidebar.write(last_update_dates.iloc[1,0], ": ", Initial_Model_Guidlines)
st.sidebar.write(last_update_dates.iloc[1,1], ": ", AI_Model) # 03-06-2021")
st.sidebar.write(last_update_dates.iloc[1,2], ": ", Hybrid_AI_Guidlines)  # 02-12-2021")
st.sidebar.write(last_update_dates.iloc[1,3], ": ", Next_Update)  # 02-12-2021")

st.sidebar.markdown("<h1 style='text-align: left; color: turquoise;'>Notice: Terms and Conditions!</h1>", unsafe_allow_html=True)
st.sidebar.write("Privacy Policy & Legal")

st.sidebar.write("Our Commitment to Privacy:  We are  collecting your answers from the screening tool to help improve the site.", 
"We collect some information for internal research only and about how you use it.",
"The information collected May or May not be personally identify you. We do not use these data to share with any third party.")

st.sidebar.write(" This is a tool for Research & Development and to be used to understand the what...if Scenarios. This tool is also complementing the AiX-Covid19 Bio-Forecasting tool")
  
    
st.sidebar.write("By entering our site, you agree Privacy and Legal terms and", 
"that we will not be liable for any harm relating to your use of our tool.",
"This tool does not provide any medical advice or recommendation and should not be used to ",
"in any form and any way for diagnose, prognosis, or treatment of any COVID19 or other medical conditions.")
    



#=============================Legal and Privacy Notice ===========
# End of file Legal Notice


st.markdown("<h1 style='text-align: left; color: turquoise;'>Privacy Policy & Legal</h1>", unsafe_allow_html=True)    
st.write("Last Data updates:")

st.write(last_update_dates.iloc[1,0], ": ", Initial_Model_Guidlines)
st.write(last_update_dates.iloc[1,1], ": ", AI_Model) # 03-06-2021")
st.write(last_update_dates.iloc[1,2], ": ", Hybrid_AI_Guidlines)  # 02-12-2021")
st.write(last_update_dates.iloc[1,3], ": ", Next_Update)  # 02-12-2021")

st.markdown("<h1 style='text-align: left; color: turquoise;'>Notice: Terms and Conditions!</h1>", unsafe_allow_html=True)
st.write("Privacy Policy & Legal")

st.write("Our Commitment to Privacy:  We are  collecting your answers from the screening tool to help improve the site.", 
    "We collect some information for internal research only and about how you use it.",
    "The information collected May or May not personally identify you. We do not use these data to share with any third party.")


st.write("This is a tool for Research & Development and to be used to understand the what...if Scenarios. This tool is also complementing the AiX-Covid19 Bio-Forecasting tool")
    
st.write("By entering our site, you agree Privacy and Legal terms and", 
    "that we will not be liable for any harm relating to your use of our tool.",
    "This tool does not provide any medical advice or recommendation and should not be used to ",
    "in any form and any way for diagnose, prognosis, or treatment of any COVID19 or other medical conditions.")    
    
    
st.write("Copyright(c) 2021 - AilluminateX LLC")        
    

st.markdown("<h1 style='text-align: left; color: turquoise;'>External Public Data - References :</h1>", unsafe_allow_html=True) 

st.write("1-CDC-Covid19 website: https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/index.html")
st.write("-CDC Forecast data and Models: https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/forecasts-cases.html")
st.write("-CDC Data Hub Ref.: https://covid19forecasthub.org/")
st.write("-CDC-Forecasting website: Both Weekly Data & Forecast Data")
st.write("-CDC-Covid19 Data also compared with processed Data by USFACTS: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/")
st.write("2- California Data: https://covid19.ca.gov/state-dashboard/")
st.write("-California Data: https://covid19.ca.gov/safer-economy/")
st.write("-California Resourecs:  https://covid19.ca.gov/data-and-tools/") 
st.write(" We have used many other public and non-public sources, to ensure we keep ourself update as far as Data & stattistics.")
st.write(" That includes both Data from Webs, News, and scientific articles")
st.write("We thanks everyone, that one way or other contributed to help to share Covid19 Data and Resources")


st.write("Copyright(c) 2021 - AilluminateX LLC")  