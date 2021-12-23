import folium
import geopandas
import pandas as pd
import streamlit as st
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

def generate_filtered_dataset(data, col, f_zipcode, f_attributes):
    f_attributes = list(set(f_attributes + ['zipcode', col]))

    if f_attributes != [] and f_zipcode != []:
        df_metrics = data.loc[data['zipcode'].isin(f_zipcode), f_attributes].copy()
    elif f_attributes == [] and f_zipcode != []:
        df_metrics = data.loc[data['zipcode'].isin(f_zipcode), :].copy()
    elif f_attributes != [] and f_zipcode == []:
        df_metrics = data.loc[:, f_attributes].copy()
    else:
        df_metrics = data.copy()
    
    if df_metrics[col].dtype in ['byte', 'int8', 'int32', 'int64']:
        df_metrics[[col, 'zipcode']].groupby('zipcode').count().reset_index()
        
    return df_metrics[[col, 'zipcode']].groupby('zipcode').mean().reset_index()

path = '../data/kc_house_data.csv'
data = get_data(path)

url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile( url )

data['price_m2'] = data['price']/(data['sqft_lot']/10.764)

st.title('House Sales Price - Data Analysis')

f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
st.write(f_attributes)

f_zipcode = st.sidebar.multiselect('Enter zipcode(s)', data['zipcode'].unique())
st.write(f_zipcode)

if f_attributes != [] and f_zipcode != []:
    df = data.loc[data['zipcode'].isin(f_zipcode), f_attributes].copy()
elif f_attributes == [] and f_zipcode != []:
    df = data.loc[data['zipcode'].isin(f_zipcode), :].copy()
elif f_attributes != [] and f_zipcode == []:
    df = data.loc[:, f_attributes].copy()
else:
    df = data.copy()

st.dataframe(df.head(20))
    
c1, c2 = st.columns((1, 1)) 
    
df1 = generate_filtered_dataset(data, 'id', f_zipcode, f_attributes)
df2 = generate_filtered_dataset(data, 'price', f_zipcode, f_attributes)
df3 = generate_filtered_dataset(data, 'sqft_living', f_zipcode, f_attributes)
df4 = generate_filtered_dataset(data, 'price_m2', f_zipcode, f_attributes)


m1 = pd.merge(df1, df2, on='zipcode', how='inner')
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
m3 = pd.merge(m2, df4, on='zipcode', how='inner')

m3.columns = ['ZIP Code', 'Total Houses', 'Price', 'Sqrt Living Room','Price/m\u00b2']

c1.header('Average by Zipcode')
c1.dataframe(m3.head(8), height=600)

c2.header( 'Descriptive Statistics' )
c2.dataframe( df.describe(), height=800 )


st.title( 'Region Overview' )

c1, c2 = st.columns(( 1, 1 ))
c1.header( 'Portfolio Density' )

df = data.sample(10)

# Base Map - Folium 
density_map = folium.Map( location=[data['lat'].mean(), 
                          data['long'].mean() ],
                          default_zoom_start=15 ) 

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

    
c2.header( 'Price Density' )

df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
df.columns = ['ZIP', 'PRICE']


geofile = geofile[geofile['ZIP'].isin( df['ZIP'].tolist() )]

region_price_map = folium.Map( location=[data['lat'].mean(), 
                               data['long'].mean() ],
                               default_zoom_start=15 ) 

region_price_map.choropleth( data = df,
                             geo_data = geofile,
                             columns=['ZIP', 'PRICE'],
                             key_on='feature.properties.ZIP',
                             fill_color='YlOrRd',
                             fill_opacity = 0.7,
                             line_opacity = 0.2,
                             legend_name='AVG PRICE' )
with c2:
    folium_static( region_price_map )
