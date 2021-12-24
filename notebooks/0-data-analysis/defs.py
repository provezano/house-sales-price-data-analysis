
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent='geoapiExercises')

def get_data(x):
    index, row = x
    
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)
    response = reverse(row['query'])
    
    place_id = response.raw['place_id']
    osm_type = response.raw['osm_type']
    country = response.raw['address']['country']
    country_code = response.raw['address']['country_code']
        
    return place_id, osm_type, country, country_code
