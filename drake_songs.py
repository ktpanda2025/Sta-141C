import pandas as pd 

df = pd.read_csv('Drake_Spotify_Data.csv')
# i want to see the column names
print(df.columns)

subset = df[['track_uri', 'energy', 'album_release_year']][df['album_release_year'] >= 2020].sort_values(by='album_release_year')
#make it order by release year ascending
#save it to a new csv file
print(subset)