import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import keys
import pandas as pd

df = pd.read_csv('Og_data/Spotify_chart_song_ranks.csv')
unique_track_uris = df['uri'].unique()
sub_list =[]

for i in range(0,len(unique_track_uris),100):
    sub_list.append(unique_track_uris[i:i+100])


client_id =keys.client_id 
client_secret=keys.client_secret

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

audio_fets = []
hhh = 0


for j in sub_list:
    audio_features = sp.audio_features(j)
    audio_fets.append(audio_features)
    print("hhhhfdhfd",hhh)
    hhh+=1



temp = []
for i in audio_fets:
    for j in i:
        temp.append(j)

audio_features_df = pd.DataFrame(temp)

audio_features_df.to_csv('Spotify_audio_features.csv', index=False)
