import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import spotipy
from spotipy import SpotifyClientCredentials
from os.path import exists
from auth import client_id, client_secret, user_id, token, playlist_id

class PullBillboard:

    #This first part takes in an input of the years you want Billboard data for and then generates a CSV file for each year
    def __init__(self):
        self.years = input("Enter the years (format: YYYY, YYYY, YYYY) you'd like Billboard top 100 data for:\n")
        self.year_list = self.years.split(', ')

    #This next part creates individual CSVs for every input year. It also compiles all of the songs into one file entitled BB_All
    def createCSV(self):
        for x in self.year_list:
            wikiurl = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_"+x
            #used https://medium.com/analytics-vidhya/web-scraping-a-wikipedia-table-into-a-dataframe-c52617e1f451
            #table_class="wikitable sortable jquery-tablesorter"
            response=requests.get(wikiurl)
            soup = BeautifulSoup(response.text, 'html.parser')
            indiatable=soup.find('table',{'class':"wikitable"})
            df = pd.read_html(str(indiatable).strip('""').replace(', ', ' '))
            df = pd.DataFrame(df[0])
            df['Title'] = df['Title'].str.replace(",","")
            if exists(f'BB{x}.csv'):
                break
            df.to_csv(f'BB{x}.csv', sep=',', index=False, encoding='utf-8-sig', quoting = csv.QUOTE_NONE)
            #can try to add in more defs to create/append csv
            if exists("BB_All.csv")==True:
                df.to_csv(f'BB_All.csv', sep=',', header = False, index=False, mode='a', encoding='utf-8-sig', quoting = csv.QUOTE_NONE)
            else:
                df.to_csv(f'BB_All.csv', sep=',', index=False, encoding='utf-8-sig', quoting = csv.QUOTE_NONE)
 
            
class GetFeatures:

    def __init__(self):
        self.user_id = user_id
        self.token = token
        self.csv = "BB_All.csv"
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        self.user_id = user_id
        self.playlist_id = playlist_id

    #Used the link from this community forum to generate playlist of this data: 
    #https://community.spotify.com/t5/App-Features/I-ve-Created-a-FREE-CSV-Playlist-Import-Tool/td-p/4979820

    #https://stackoverflow.com/questions/39086287/spotipy-how-to-read-more-than-100-tracks-from-a-playlist
    #This next part is directly from the link above
    def get_playlist_tracks_more_than_100_songs(self):
        results = self.sp.user_playlist_tracks(self.user_id, self.playlist_id)
        tracks = results['items']
        while results['next']:
            results = self.sp.next(results)
            tracks.extend(results['items'])
        results = tracks    

        playlist_tracks_id = []
        playlist_tracks_titles = []
        playlist_tracks_artists = []
        playlist_tracks_first_artists = []
        playlist_tracks_first_release_date = []
        playlist_tracks_popularity = []

        for i in range(len(results)):
            #print(i) # Counter
            if i == 0:
                playlist_tracks_id = results[i]['track']['id']
                playlist_tracks_titles = results[i]['track']['name']
                playlist_tracks_first_release_date = results[i]['track']['album']['release_date']
                playlist_tracks_popularity = results[i]['track']['popularity']

                artist_list = []
                for artist in results[i]['track']['artists']:
                    artist_list= artist['name']
                playlist_tracks_artists = artist_list

                features = self.sp.audio_features(playlist_tracks_id)
                features_df = pd.DataFrame(data=features, columns=features[0].keys())
                features_df['title'] = playlist_tracks_titles
                features_df['all_artists'] = playlist_tracks_artists
                features_df['popularity'] = playlist_tracks_popularity
                features_df['release_date'] = playlist_tracks_first_release_date
                features_df = features_df[['id', 'title', 'all_artists', 'popularity', 'release_date',
                                        'danceability', 'energy', 'key', 'loudness',
                                        'mode', 'acousticness', 'instrumentalness',
                                        'liveness', 'valence', 'tempo',
                                        'duration_ms', 'time_signature']]
                continue
            else:
                try:
                    playlist_tracks_id = results[i]['track']['id']
                    playlist_tracks_titles = results[i]['track']['name']
                    playlist_tracks_first_release_date = results[i]['track']['album']['release_date']
                    playlist_tracks_popularity = results[i]['track']['popularity']
                    artist_list = []
                    for artist in results[i]['track']['artists']:
                        artist_list= artist['name']
                    playlist_tracks_artists = artist_list
                    features = self.sp.audio_features(playlist_tracks_id)
                    new_row = {'id':[playlist_tracks_id],
                'title':[playlist_tracks_titles],
                'all_artists':[playlist_tracks_artists],
                'popularity':[playlist_tracks_popularity],
                'release_date':[playlist_tracks_first_release_date],
                'danceability':[features[0]['danceability']],
                'energy':[features[0]['energy']],
                'key':[features[0]['key']],
                'loudness':[features[0]['loudness']],
                'mode':[features[0]['mode']],
                'acousticness':[features[0]['acousticness']],
                'instrumentalness':[features[0]['instrumentalness']],
                'liveness':[features[0]['liveness']],
                'valence':[features[0]['valence']],
                'tempo':[features[0]['tempo']],
                'duration_ms':[features[0]['duration_ms']],
                'time_signature':[features[0]['time_signature']]
                }

                    dfs = [features_df, pd.DataFrame(new_row)]
                    features_df = pd.concat(dfs, ignore_index = True)
                except:
                    continue       
        return features_df
    
if __name__ == '__main__':
    bb = PullBillboard()
    bb.createCSV()
    gen = GetFeatures()
    features_df=gen.get_playlist_tracks_more_than_100_songs()
    #after creating the csv, I went through and manually edited
    #the file. The escape char and general csv format had
    #issues that caused some rows to shift a few columns over
    features_df.to_csv('BB_Features.csv', sep=',', index=False, encoding='utf-8-sig', quoting = csv.QUOTE_NONE, escapechar='\\')