import folium
import geopandas
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(layout='wide')

@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    return data

@st.cache(allow_output_mutation=True)
def get_geofile( url ):
    geofile = geopandas.read_file( url )
    return geofile

def filter_dataframe(data, f_zipcode, f_attributes):
    if f_attributes != [] and f_attributes not in f_attributes:
        f_attributes = list(dict.fromkeys(['zipcode'] + f_attributes))
    
    if f_attributes != [] and f_zipcode != []:
        df_metrics = data.loc[data['zipcode'].isin(f_zipcode), f_attributes].copy()
    elif f_attributes == [] and f_zipcode != []:
        df_metrics = data.loc[data['zipcode'].isin(f_zipcode), :].copy()
    elif f_attributes != [] and f_zipcode == []:
        df_metrics = data.loc[:, f_attributes].copy()
    else:
        df_metrics = data.copy()

    return df_metrics

def generate_filtered_dataset(data, col, f_zipcode, f_attributes):
    if f_attributes != [] and f_attributes not in f_attributes:
        f_attributes = list(dict.fromkeys([col] + f_attributes))
    
    df_metrics = filter_dataframe(data, f_zipcode, f_attributes)
    
    if df_metrics[col].dtype in ['byte', 'int8', 'int32', 'int64']:
        df_metrics[[col, 'zipcode']].groupby('zipcode').count().reset_index()
        
    return df_metrics[[col, 'zipcode']].groupby('zipcode').mean().reset_index()


# Data Ingest
path = 'kc_house_data.csv'
data = get_data(path)

# Get Zip Codes Data from ArcGis
#url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
url = 'Zip_Codes.geojson'
geofile = get_geofile( url )

# Feature Engineering
data['price_m2'] = data['price']/(data['sqft_lot']/10.764)
data['date'] = pd.to_datetime(data['date'])
data['yr_built'] = data['yr_built'].astype(int)

# Add image to sidebar
st.sidebar.image("https://cdn4.iconfinder.com/data/icons/02-real-estate-outline-colors/91/RealEstate-64-512.png", use_column_width=True)

# Add title to sidebar
st.sidebar.title('Table filters')

# Add filtering options to the dataframe
f_attributes = st.sidebar.multiselect('Enter columns', [col for col in data.columns if col != 'zipcode'])
f_zipcode = st.sidebar.multiselect('Enter zipcode(s)', np.sort(data['zipcode'].unique()))

# Apply the filters
df = filter_dataframe(data, f_zipcode, f_attributes)

# Main page title
st.title('House sales price - Data analysis')

# Header to the Dataframe
st.header('Data Sample (showing only a sample of 20 houses)')

#Show the top 20 of our dataframe
st.dataframe(df.head(20))

# Divide main page in two columns
c1, c2 = st.columns((1, 1)) 

# Generate filtered data to apply some stats on
df1 = generate_filtered_dataset(data, 'id', f_zipcode, f_attributes)
df2 = generate_filtered_dataset(data, 'price', f_zipcode, f_attributes)
df3 = generate_filtered_dataset(data, 'sqft_living', f_zipcode, f_attributes)
df4 = generate_filtered_dataset(data, 'price_m2', f_zipcode, f_attributes)

m1 = pd.merge(df1, df2, on='zipcode', how='inner')
m2 = pd.merge(m1,  df3, on='zipcode', how='inner')
m3 = pd.merge(m2,  df4, on='zipcode', how='inner')

m3.columns = ['ZIP code', 'Total houses', 'price', 'SQRT living','Price/m\u00b2']

c1.header('Average per zipcode')
c1.dataframe(m3.head(8), height=600)

c2.header('Descriptive statistics')
c2.dataframe(df.describe(include=['float', 'int']), height=800)

st.title('Region overview')

c1, c2 = st.columns(( 1, 1 ))
c1.header('Portfolio density')

df = data.sample(50)
 
density_map = folium.Map( location=[data['lat'].mean(), 
                          data['long'].mean() ],
                          default_zoom_start=15) 

marker_cluster = MarkerCluster().add_to( density_map )
for name, row in df.iterrows():
    folium.Marker( [row['lat'], row['long'] ], 
        popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( row['price'],
                                     row['date'],
                                     row['sqft_living'],
                                     row['bedrooms'],
                                     row['bathrooms'],
                                     row['yr_built'] ) ).add_to( marker_cluster )
with c1:
    folium_static( density_map )

    
c2.header('Price density')

df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
df.columns = ['ZIP', 'PRICE']


geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

region_price_map = folium.Map(location=[data['lat'].mean(), 
                               data['long'].mean()],
                               default_zoom_start=15) 

region_price_map.choropleth(data = df,
                             geo_data = geofile,
                             columns=['ZIP', 'PRICE'],
                             key_on='feature.properties.ZIP',
                             fill_color='YlOrRd',
                             fill_opacity = 0.7,
                             line_opacity = 0.2,
                             legend_name='AVG PRICE')
with c2:
    folium_static(region_price_map)
    
st.sidebar.title('Commercial options')
st.title('Commercial attributes')

min_year_built = int(data['yr_built'].min())
max_year_built = int(data['yr_built'].max())+1
mean_year_built = int(data['yr_built'].mean())

st.sidebar.subheader('Select Max Year Built')
f_yr_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, mean_year_built)

st.header('Average Price Per Year Built')
df = data[data['yr_built']<f_yr_built]
df = df[['price', 'yr_built']].groupby('yr_built').mean().reset_index()

df = data[['price', 'yr_built']].groupby( 'yr_built' ).mean().reset_index()
fig = px.line(df, x='yr_built', y='price')

st.plotly_chart(fig, use_container_width=True)

min_date = datetime.strptime(datetime.strftime(data['date'].min(), '%Y-%m-%d'), '%Y-%m-%d')
max_date = datetime.strptime(datetime.strftime(data['date'].max(), '%Y-%m-%d'), '%Y-%m-%d')
mode_date = datetime.strptime(datetime.strftime(data['date'].mode()[0], '%Y-%m-%d'), '%Y-%m-%d')

st.sidebar.subheader('Select max date')
print(min_date, type(min_date))
f_date = st.sidebar.slider('Date', min_date, max_date, mode_date)

st.header('Average price per date')
df = data[data['date']<f_date]
df = df[['price', 'date']].groupby('date').mean().reset_index()

fig = px.line(df, x='date', y='price')

st.plotly_chart(fig, use_container_width=True)

st.header('Price Distribution')
st.sidebar.subheader('Select Max Price')

min_price = int(data['price'].min())
max_price = int(data['price'].max())+1
mean_price = int(data['price'].mean())

f_price=st.sidebar.slider('Price', min_price, max_price, mean_price)

df = data[data['price']<f_price]
fig = px.histogram(df, x='price', nbins=50)

st.plotly_chart(fig, use_container_width=True)

st.title('House attributes')
c1, c2 = st.columns((1, 1))

st.sidebar.title('House attributes options')

f_bedrooms=st.sidebar.selectbox('Maximum number of bedrooms', np.sort(data['bedrooms'].unique()), index=4)
df = data[data['bedrooms']<=f_bedrooms]
c1.header('# of houses given a maximum of bedrooms')
fig = px.histogram(df, x='bedrooms', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

f_bathrooms=st.sidebar.selectbox('Maximum number of bathrooms', np.sort(data['bathrooms'].unique()), index=2)
df = data[data['bathrooms']<=f_bathrooms]
c1.header('# of houses given a maximum of bathrooms')
fig = px.histogram(data, x='bathrooms', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

f_floors=st.sidebar.selectbox('Maximum number of floors', np.sort(data['floors'].unique()), index=2)
df = data[data['floors']<=f_floors]
c2.header('# of houses given a maximum of floors')
fig = px.histogram(data, x='floors', nbins=19)
c2.plotly_chart(fig, use_container_width=True)

f_waterfront=st.sidebar.selectbox('Maximum number of waterfront', np.sort(data['waterfront'].unique()), index=0)
df = data[data['waterfront']<=f_waterfront]
c2.header('# of houses given a maximum of waterfront')
fig = px.histogram(data, x='waterfront', nbins=19)
c2.plotly_chart(fig, use_container_width=True)
