# Pop Music Trend Analysis
*This project utilizes the Billboard 100 and Spotify's API*  

## Introduction
The purpose of this project was to analyze music trends throughout the decades.
 For my purposes, I specifically looked at music from 1980 to 2020 using 5 year intervals.
 Because the Spotify API can only pull features for 100 songs at a time and because I wanted
 to cross check my findings, the BilllBoard Data folder contains a CSV with all of the songs
 analyzed as well as all of the songs separated by year.

## Configuration and Installation
To utilize this code, you need a working Spotify account. The authentication key can be requested here: https://developer.spotify.com/documentation/general/guides/authorization/app-settings/
This page explains the API in greater detail: https://developer.spotify.com/documentation/general/guides/authorization/code-flow/

From there, use the auth.py file to gain access to the song features. This part is essential for accessing the rest of the code. Input your auth. key in the appropriate place as well as any other required information.

Then, use the BBWS.py file to get the necessary CSV data of your songs of interest. Follow the instructions within the code. From there, you can use the graphs.py file to analyze the data via unsupervised machine learning.

## Contact Information
If you have any questions or suggestions for this project, please feel free to contact me at shivanip098@gmail.com !

## Challenges
While most of the data can be quickly processed, there is a chance that some of the CSV files will need to be manually cleaned. I found that the song data occasionally needed to be reformatted, but this was only an issue for <1% of the songs I encountered.

## Credits
A HUGE Thank you to Spotify user nick_wanders for creating this CSV Playlist Import tool, allowing me to pull features from specific playlists! I've linked his post in the BBWS.py file.
