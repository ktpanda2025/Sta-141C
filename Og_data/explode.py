import pandas as pd 
df = pd.read_csv('Spotify_chart_song_ranks.csv')

new = df.assign(artist_names = df.artist_names.str.split(',')).explode('artist_names')
# expor to csv
new.to_csv('New_Spspotify_chart_song_ranks.csv', index=False)