import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from numerize import numerize
import warnings
warnings.filterwarnings('ignore')


#-----Start of Set Up-----#

st.set_page_config(page_title='ITZY', page_icon = '', initial_sidebar_state = 'auto',layout="wide")

my_page = st.sidebar.radio('Contents',['About ITZY',"Exploring ITZY's Spotify Data",'Widen ITZY\'s Listenership','Spotlighting ITZY', 'Recommendations','The Team']) # creates sidebar #

#-----End of Set Up-----#


#-----Start of Page 1 (About ITZY)-----#

if my_page == 'About ITZY':

    st.title("Making ITZY the Next Biggest K-Pop Girl Group")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) #space
    st.image('images/ITZY.png',use_column_width=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) #space
    st.markdown("Our group’s hypothetical client is ITZY’s management team. In this project, we wanted to help them with strategizing on how to make ITZY the next biggest K-Pop girl group. The ways we thought of doing that is by (1) widening their listenership and (2) creating a spotlight on ITZY.")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) #space
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) #space
    st.markdown('<div style="text-align: left;font-size: 20px;font-weight: bold;">Who is ITZY?</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) #space
    a1,a2,a3 = st.beta_columns((4,1,4))
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) #space
    a1.markdown('<iframe src="https://open.spotify.com/embed/artist/2KC9Qb60EaY0kW4eH68vr3" width=100% height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',unsafe_allow_html=True)
    a1.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) #space
    a3.markdown(" ITZY is a South Korean girl group formed by JYP Entertainment who debuted on February 2019. They currently have an almost 3.5 million monthly listeners on Spotify with already 3 EPs, 1 album, and 1 single album under their belt. They’re popularly known for their own brand of spunkiness and their relentless advocacy for being true to one’s self.")
    
    
#-----End of Page 1 (About ITZY)-----#

#-----Start of Page 2 (Exploring ITZY's Spotify Data)-----#

if my_page == "Exploring ITZY's Spotify Data":
    
    @st.cache
    def load_data(URL):
        data = pd.read_csv(URL)
        return data

    df1 = load_data("datasets/kpop_girlgroup_data.csv")
    df2 = load_data("datasets/song characteristics.csv")
    df0 = load_data("datasets/timeline.csv")

    st.title("Exploring ITZY's Spotify Data")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    st.markdown('The data we used for our exploratory data analysis (EDA) were scraped through Spotify API and are the daily top 200  most streamed tracks from January 1, 2017 to January 15, 2021. We then focused our analysis on K-Pop girl groups and on ITZY.',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    st.markdown('<div style="font-size: 25px;font-weight: bold;">ITZY is the 5th most streamed K-Pop girl group</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: gray; font-size: 18px;">With 23M total streams for their charting songs</div>',unsafe_allow_html=True)
    top5 = df1.groupby('artist')[['streams']].sum().sort_values(by="streams", ascending=False).head(5).reset_index()
    top5.sort_values(by="streams", ascending=True, inplace=True)
    
    fig = px.bar(top5, x="streams", y="artist", orientation='h', height=350,
                 text=top5["streams"].apply(lambda x: numerize.numerize(x)))
    
    colors = ['#E0B336'] * 5
    colors[0] = '#B88F89'

    fig.update_traces(marker=dict(color=colors), textposition='outside',
                      textfont=dict(size=14, color=colors), width = 0.65)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False,
                      xaxis = {'title': 'Total Streams for Charting Songs', 'range': [0,300000000],
                               'showline':True, 'linecolor':'#999999', 'tickfont':dict(color= '#999999'),
                               'showgrid' : False,'fixedrange':True,'zeroline': False,
                               'titlefont' : dict(color = "#999999", size = 16)},
                      yaxis = {'title': '', 'showgrid' : False, 'fixedrange':True},
                      margin=dict(l=0, r=0, b=15, t=25, pad=15), font=dict(size=14)
                     )
    
    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("Despite being new to the game, ITZY is currently the 5th most streamed K-pop girl group on Spotify with almost 23 million total streams for their charting songs. They’re accompanied in the top 5 by BLACKPINK, TWICE, MOMOLAND, and Red Velvet.")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    st.markdown('<div style="text-align: left;font-size: 25px;font-weight: bold;">Streams for ITZY’s charting songs increase with every EP/album release</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: gray; font-size: 16px;font-style: italic;">Hover over the plotted line to view the number of streams on a given day</div>',unsafe_allow_html=True)
    dates = ["2019-02-13","2019-07-29","2020-03-09","2020-08-17"]
    plots = df0[(df0['date'] == dates[0]) | (df0['date'] == dates[1]) | (df0['date'] == dates[2]) | (df0['date'] == dates[3])]
    plots['album'] = ['It\'z Different', 'It\'z ICY', 'It\'z Me', 'Not Shy']

    fig = px.line(df0, x='date', y="streams")

    fig.update_traces(line=dict(color='#B88F89', width=2),
                      hovertemplate = "<br>".join(["%{x}", "%{y} streams<br><extra></extra>"]))

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      xaxis = {'title': '', 'titlefont' : dict(size = 16), 'fixedrange':True,
                               'gridcolor':'#D9D9D9'},
                      yaxis = {'title': 'Number of Streams','titlefont' : dict(size = 16),
                               'fixedrange':True, 'gridcolor':'#D9D9D9', 'zerolinecolor':'#D9D9D9'},
                      margin=dict(l=5, r=5, b=5, t=25, pad=15), font=dict(size=14),
                      hoverlabel=dict(bgcolor="#FAFAFA", font_size=15)
                         )

    fig.add_scatter(x = plots.date, y = plots.streams, mode = 'markers + text', marker = {'color':'#424242', 'size':10},
                    showlegend = False, text = plots['album'], textposition='bottom center',
                    hovertemplate = "<br>".join(["%{text}","%{x}", "%{y} streams<br><extra></extra>"]))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("Through our further EDA, we saw that the streams for ITZY’s charting songs increase with every EP or album release during the dates of 12 February 2019, 29 July 2019, 9 March 2020, and 17 August 2020. That got us wondering what is it about ITZY’s songs that made people want to listen to them? What’s their difference with the other K-Pop girl groups mentioned earlier?")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    st.markdown('<div style="text-align: left;font-size: 25px;font-weight: bold;">ITZY has the most danceable charting songs</div>',unsafe_allow_html=True)
    
    dance = df2[['artist','danceability']].sort_values(by="danceability", ascending=True)
    
    fig = px.bar(dance, x="danceability", y="artist", orientation='h', height=350,
                 text=dance["danceability"].apply(lambda x: '{0:1.2f}'.format(x)))

    fig.update_traces(marker=dict(color=['#d9d9d9','#d9d9d9','#d9d9d9','#d9d9d9','#B88F89']), textposition='outside',
                      textfont=dict(size=14, color=['#999999','#999999','#999999','#999999','#B88F89']), width = 0.65)


    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False,
                      xaxis = {'title': 'Average Danceability of Charting Songs', 'range': [0, 1],
                               'showticklabels' : True, 'showgrid' : False,'zeroline': False,
                               'titlefont' : dict(color = "#B88F89", size = 16), 'fixedrange':True,
                              'showline':True, 'linecolor':'#D9D9D9',},
                      yaxis = {'title': '', 'showgrid' : False,'zeroline': False, 'fixedrange':True},
                      margin=dict(l=0, r=0, b=0, t=25, pad=15), font=dict(size=14)
                     )
    
    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("In our exploration of the Spotify data, we saw that compared to the other K-Pop girl groups, ITZY has the most danceable charting songs. The **danceability** audio feature of Spotify tracks *describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.*")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    st.markdown('<div style="text-align: left;font-size: 25px;font-weight: bold;">ITZY has the most energetic charting songs</div>',unsafe_allow_html=True)

    energy = df2[['artist','energy']].sort_values(by="energy", ascending=True)

    fig = px.bar(energy, x="energy", y="artist", orientation='h', height=350,
                 text=energy["energy"].apply(lambda x: '{0:1.2f}'.format(x)))
    
    colors = ['#E0B336'] * 5
    colors[4] = '#B88F89'

    fig.update_traces(marker=dict(color=colors), textposition='outside',
                      textfont=dict(size=14,color=colors), width = 0.65)


    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False,
                      xaxis = {'title': 'Average Energy of Charting Songs', 'range': [0, 1],
                               'showgrid' : False,'zeroline': False,'fixedrange':True,
                               'showline':True, 'linecolor':'#999999', 'tickfont':dict(color= '#999999'),
                               'titlefont' : dict(color = "#999999", size = 16)},
                      yaxis = {'title': '', 'showgrid' : False,'zeroline': False, 'fixedrange':True},
                      margin=dict(l=0, r=0, b=0, t=25, pad=15), font=dict(size=14),
                     )
    
    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('ITZY also has the most energetic charting songs among the K-Pop girl groups mentioned earlier, with TWICE being not so far behind on second place. The **energy** audio feature of Spotify tracks *represents a perceptual measure of intensity and activity. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.* A value of 0.0 means the song has low energy and 1.0  means the song has high energy.')
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    st.markdown('<div style="text-align: left;font-size: 25px;font-weight: bold;">ITZY’s  charting songs tend to have a positive sound</div>',unsafe_allow_html=True)
   
    valence = df2[['artist','valence']].sort_values(by="valence", ascending=True)

    fig = px.bar(valence, x="valence", y="artist", orientation='h', height=350,
                 text=valence["valence"].apply(lambda x: '{0:1.2f}'.format(x)))
    
    colors = ['#E0B336'] * 5
    colors[2] = '#B88F89'

    fig.update_traces(marker=dict(color=colors), textposition='outside',
                      textfont=dict(size=14, color=colors), width = 0.65)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False,
                      xaxis = {'title': 'Average Valence of Charting Songs', 'range': [0, 1],
                               'showgrid' : False,'zeroline': False, 'fixedrange':True,
                               'showline':True, 'linecolor':'#999999', 'tickfont':dict(color= '#999999'),
                               'titlefont' : dict(color = "#999999", size = 16)},
                      yaxis = {'title': '', 'showgrid' : False,'zeroline': False, 'fixedrange':True},
                      margin=dict(l=0, r=0, b=0, t=25, pad=15), font=dict(size=14)
                     )
    
    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("In terms of how positive their tracks sound, ITZY scored a little more than halfway—still sounds a bit positive but not as cheerful as that of MOMOLAND's. The **valence** audio feature of Spotify tracks *describes the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)*.")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="font-size: large;font-weight: bold;">Note</div>',unsafe_allow_html=True)
    st.markdown("Italicized parts are directly lifted from Spotify's Web API Reference for developers.")
    st.markdown("""<a style='display: block; color:#77A9B4; text-decoration: none;' href="https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-features"> Access the reference through this link.</a>""", unsafe_allow_html=True)
    
#-----End of Page 2 (Exploring ITZY's Spotify Data)-----#


#-----Start of Page 3 (Widen ITZY's Listenership)-----#

elif my_page == 'Widen ITZY\'s Listenership':
    
    @st.cache
    def load_data(URL):
        data = pd.read_csv(URL)
        return data
    
    df3 = load_data("datasets/popreco_streams.csv")
    df3.sort_values(by='streams', ascending=True,inplace=True)
    df4 = load_data("datasets/rnbreco_streams.csv")
    df4.sort_values(by='streams', ascending=True,inplace=True)
    df5 = load_data("datasets/rapreco_streams.csv")
    df5.sort_values(by='streams', ascending=True,inplace=True)
    
    st.title("Widen listenership: Collaborate with other artists")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.image('images/widen listenership.png',use_column_width=True)
    st.markdown("Our first suggestion to ITZY’s management team is to widen the group's listenership by collaborating with other artists. According to a study by Ordanini et. al entitled *\"The featuring phenomenon in music: how combining artists of different genres increases a song’s popularity\"* (2018), the average likelihood of entering in Billboards’ Top 10 Hit by a song with a featured artist is 18.4%, significantly greater than the 13.9% likelihood for songs that do not include a featured artist.")
    st.markdown("")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    st.markdown('<div style="text-align: left;font-size: 25px;font-weight: bold;">Who Should ITZY Collaborate With?</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("To look for possible music collaborators for ITZY, we’ve built a recommendation engine with the following technical details and resulting metrics:",unsafe_allow_html=True)
    st.markdown('---')
    c1,c2 = st.beta_columns((1,2))
    c1.markdown('<div style="text-align: left;color: #424242; font-weight: bold;">Classification algorithm used </div>',unsafe_allow_html=True)
    c2.markdown("k Nearest Neighbors",unsafe_allow_html=True)
 
    c3,c4 = st.beta_columns((1,2))
    c3.markdown('<div style="text-align: left;font-weight: bold;">Optimal k</div>',unsafe_allow_html=True)
    c4.markdown("19",unsafe_allow_html=True)

    c5,c6 = st.beta_columns((1,2))
    c5.markdown('<div style="text-align: left;font-weight: bold;">Genres used for training</div>',unsafe_allow_html=True)
    c6.markdown("Pop, R&B, Rap",unsafe_allow_html=True)

    c7,c8 = st.beta_columns((1,2))
    c7.markdown('<div style="text-align: left;font-weight: bold;">Features used to train model</div>',unsafe_allow_html=True)
    c8.markdown("danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo",unsafe_allow_html=True)

    c9,c10 = st.beta_columns((1,2))
    c9.markdown('<div style="text-align: left;font-weight: bold;">Target variable</div>',unsafe_allow_html=True)
    c10.markdown("Genre",unsafe_allow_html=True)

    c11,c12 = st.beta_columns((1,2))
    c11.markdown('<div style="text-align: left;font-weight: bold;">Metrics</div>',unsafe_allow_html=True)
    c12.markdown("*Accuracy:* 65%", unsafe_allow_html=True)
    c12.markdown("*Multiclass ROC AUC:* 80% ", unsafe_allow_html=True)
    c12.markdown("*Multiclass F1:* 64%", unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("The data we used to train the model were scraped from the Spotify API and are tracks from playlists with titles explicitly stating the genre (Pop, R&B, Rap). We took the playlist with the most number of followers with the assumption that since lots of people follow those playlists, it means they agree that the tracks on those playlists belong on the genres stated on their titles.", unsafe_allow_html=True)
    st.markdown("After training the model to classify the genres, a representative track was made by aggregating the audio features of ITZY’s charting songs. Its similarity was then computed against other songs based on the cosine distance of the audio features and the predicted genre probabilities. A playlist with similar audio features as that of ITZY’s representative track is deployed on Spotify and is shown below:", unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<iframe src="https://open.spotify.com/embed/playlist/3iwKMiQ0EVEFooxMwGDHjb" width=100% height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left;font-size: 25px;font-weight: bold;">List of Recommended Artists per Genre</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("For each genre, we picked five songs with the greatest similarity to ITZY's representative track. We interpreted that those artists whose songs have great similarity with the representative track are possible collaborators.", unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('We also wanted to know how the recommended artists fare when it comes to the number of streams of their charting songs. ITZY’s team might also consider going for an artist with high number of streams when deciding who to collaborate with.',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('---')
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 20px;">Recommended Pop Artists</div>',unsafe_allow_html=True)
    st.markdown('<div style="text-align: left; color: gray; font-size: 17px;">Arranged from lowest to highest cosine distance</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    #Pop reco
    a1,a2,a3,a4,a5 = st.beta_columns(5)
    a1.image('images/pop/Sigma.jpg',use_column_width=True)
    a1.markdown('<div style="text-align: center;color: #c6793a; font-weight: bold;">Sigma</div>',unsafe_allow_html=True)
    a1.markdown('<div style="text-align: center;color: #666666; font-style: italic">3.2*</div>',unsafe_allow_html=True)
    a1.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    a2.image('images/pop/Lady Gaga.jpg',use_column_width=True)
    a2.markdown('<div style="text-align: center;color: #c6793a; font-weight: bold;">Lady Gaga</div>',unsafe_allow_html=True)
    a2.markdown('<div style="text-align: center;color: #666666; font-style: italic">3.26*</div>',unsafe_allow_html=True)
    a2.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    a3.image('images/pop/Flo Rida.jpg',use_column_width=True)
    a3.markdown('<div style="text-align: center;color: #c6793a; font-weight: bold;">Flo Rida</div>',unsafe_allow_html=True)
    a3.markdown('<div style="text-align: center;color: #666666; font-style: italic">4.5*</div>',unsafe_allow_html=True)
    a3.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    a4.image('images/pop/Nevada.jpg',use_column_width=True)
    a4.markdown('<div style="text-align: center;color: #c6793a; font-weight: bold;">Nevada</div>',unsafe_allow_html=True)
    a4.markdown('<div style="text-align: center;color: #666666; font-style: italic">4.6*</div>',unsafe_allow_html=True)
    a4.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    a5.image('images/pop/Camila.jpg',use_column_width=True)
    a5.markdown('<div style="text-align: center;color: #c6793a; font-weight: bold;">Camila Cabello</div>',unsafe_allow_html=True)
    a5.markdown('<div style="text-align: center;color: #666666; font-style: italic">4.67*</div>',unsafe_allow_html=True)
    a5.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("*\* Cosine distance value (×10^-3). Lower value indicates closeness of the audio features of the recommended artist's song to ITZY's representative track.*",unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 20px;">Camila Cabello is the most streamed recommended pop artist</div>',unsafe_allow_html=True)
    st.markdown('<div style="text-align: left; color: gray; font-size: 17px;">Based on her charting songs for the period of January 2017 to January 2021</div>',unsafe_allow_html=True)

    fig = px.bar(df3, x="streams", y="artist", orientation='h', height=350,
                 text=df3["streams"].apply(lambda x: numerize.numerize(x)))

    fig.update_traces(marker=dict(color=['#C6793A']*5), textposition='outside',
                      textfont=dict(size=14, color=['#C6793A']*5), width = 0.65)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False,
                      xaxis = {'title': 'Total Streams for Charting Songs', 'range': [0, 110000000],
                               'showgrid' : False, 'zeroline': False,'fixedrange':True,
                               'titlefont' : dict(color = "#999999", size = 16),
                              'showline':True, 'linecolor':'#999999', 'tickfont':dict(color= '#999999')},
                      yaxis = {'title': '', 'showgrid' : False,'zeroline': False, 'fixedrange':True},
                      margin=dict(l=0, r=0, b=10,t=25, pad=15), font=dict(size=14)
                     )

    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    st.markdown('---')
        
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 20px;">Recommended R&B Artists</div>',unsafe_allow_html=True)
    st.markdown('<div style="text-align: left; color: gray; font-size: 17px;">Arranged from lowest to highest cosine distance</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    #R&B Reco
    b1,b2,b3,b4,b5 = st.beta_columns(5)
    b1.image('images/rnb/Drake.jpg',use_column_width=True)
    b1.markdown('<div style="text-align: center;color: #77a9b4; font-weight: bold;">Drake</div>',unsafe_allow_html=True)
    b1.markdown('<div style="text-align: center;color: #666666; font-style: italic">86.39*</div>',unsafe_allow_html=True)
    b1.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    b2.image('images/rnb/Charlie.jpg',use_column_width=True)
    b2.markdown('<div style="text-align: center;color: #77a9b4; font-weight: bold;">Charlie Puth</div>',unsafe_allow_html=True)
    b2.markdown('<div style="text-align: center;color: #666666; font-style: italic">89.57*</div>',unsafe_allow_html=True)
    b2.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    b3.image('images/rnb/Why.jpg',use_column_width=True)
    b3.markdown('<div style="text-align: center;color: #77a9b4; font-weight: bold;">Why Don’t We</div>',unsafe_allow_html=True)
    b3.markdown('<div style="text-align: center;color: #666666; font-style: italic">91.11*</div>',unsafe_allow_html=True)
    b3.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    b4.image('images/rnb/Kygo.jpg',use_column_width=True)
    b4.markdown('<div style="text-align: center;color: #77a9b4; font-weight: bold;">Kygo</div>',unsafe_allow_html=True)
    b4.markdown('<div style="text-align: center;color: #666666; font-style: italic">91.67*</div>',unsafe_allow_html=True)
    b4.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    b5.image('images/rnb/David.jpg',use_column_width=True)
    b5.markdown('<div style="text-align: center;color: #77a9b4; font-weight: bold;">David Guetta</div>',unsafe_allow_html=True)
    b5.markdown('<div style="text-align: center;color: #666666; font-style: italic">93.02*</div>',unsafe_allow_html=True)
    b5.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("*\* Cosine distance value (×10^-3). Lower value indicates closeness of the audio features of the recommended artist's song to ITZY's representative track.*",unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 20px;">Drake is the most streamed recommended R&B artist</div>',unsafe_allow_html=True)
    st.markdown('<div style="text-align: left; color: gray; font-size: 17px;">Based on his charting songs for the period of January 2017 to January 2021</div>',unsafe_allow_html=True)

    fig = px.bar(df4, x="streams", y="artist", orientation='h', height=350,
                 text=df3["streams"].apply(lambda x: numerize.numerize(x)))

    fig.update_traces(marker=dict(color=['#77A9B4']*5), textposition='outside',
                      textfont=dict(size=14, color=['#77A9B4']*5), width = 0.65)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False,
                      xaxis = {'title': 'Total Streams for Charting Songs', 'range': [0, 110000000],
                               'showgrid' : False, 'zeroline': False, 'fixedrange':True,
                               'showline':True, 'linecolor':'#999999', 'tickfont':dict(color= '#999999'),
                               'titlefont' : dict(color = "#999999", size = 16)},
                      yaxis = {'title': '', 'showgrid' : False,'zeroline': False, 'fixedrange':True},
                      margin=dict(l=0, r=0, b=0,t=25, pad=15), font=dict(size=14)
                     )

    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    st.markdown('---')
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 20px;">Recommended Rap Artists</div>',unsafe_allow_html=True)
    st.markdown('<div style="text-align: left; color: gray; font-size: 17px;">Arranged from lowest to highest cosine distance</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    #Rap Reco
    d1,d2,d3,d4,d5 = st.beta_columns(5)
    d1.image('images/rap/Lost Kings.jpg',use_column_width=True)
    d1.markdown('<div style="text-align: center;color: #565b7b; font-weight: bold;">Lost Kings</div>',unsafe_allow_html=True)
    d1.markdown('<div style="text-align: center;color: #666666; font-style: italic">85.18*</div>',unsafe_allow_html=True)
    d1.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    d2.image('images/rap/Charlie Puth.jpg',use_column_width=True)
    d2.markdown('<div style="text-align: center;color: #565b7b; font-weight: bold;">Charlie Puth</div>',unsafe_allow_html=True)
    d2.markdown('<div style="text-align: center;color: #666666; font-style: italic">91.89*</div>',unsafe_allow_html=True)
    d2.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    d3.image('images/rap/Lil Nas X.jpg',use_column_width=True)
    d3.markdown('<div style="text-align: center;color: #565b7b; font-weight: bold;">Lil Nas X</div>',unsafe_allow_html=True)
    d3.markdown('<div style="text-align: center;color: #666666; font-style: italic">93.54*</div>',unsafe_allow_html=True)
    d3.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    d4.image('images/rap/Tyga.jpg',use_column_width=True)
    d4.markdown('<div style="text-align: center;color: #565b7b; font-weight: bold;">Tyga</div>',unsafe_allow_html=True)
    d4.markdown('<div style="text-align: center;color: #666666; font-style: italic">93.65*</div>',unsafe_allow_html=True)
    d4.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    d5.image('images/rap/Demi.jpg',use_column_width=True)
    d5.markdown('<div style="text-align: center;color: #565b7b; font-weight: bold;">Demi Lovato</div>',unsafe_allow_html=True)
    d5.markdown('<div style="text-align: center;color: #666666; font-style: italic">97.21*</div>',unsafe_allow_html=True)
    d5.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("*\* Cosine distance value (×10^-3). Lower value indicates closeness of the audio features of the recommended artist's song to ITZY's representative track.*",unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 20px;">Charlie Puth is the most streamed recommended rap artist</div>',unsafe_allow_html=True)
    st.markdown('<div style="text-align: left; color: gray; font-size: 17px;">Based on his charting songs for the period of January 2017 to January 2021</div>',unsafe_allow_html=True)

    fig = px.bar(df4, x="streams", y="artist", orientation='h', height=350,
                 text=df3["streams"].apply(lambda x: numerize.numerize(x)))

    fig.update_traces(marker=dict(color=['#565B7B']*5), textposition='outside',
                      textfont=dict(size=14, color=['#565B7B']*5),width = 0.65)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False,
                      xaxis = {'title': 'Total Streams for Charting Songs', 'range': [0, 110000000],
                               'showgrid' : False, 'zeroline': False, 'fixedrange':True,
                               'titlefont' : dict(color = "#999999", size = 16),
                               'showline':True, 'linecolor':'#999999', 'tickfont':dict(color= '#999999')},
                      yaxis = {'title': '', 'showgrid' : False,'zeroline': False, 'fixedrange':True},
                      margin=dict(l=0, r=0, b=10,t=25, pad=15), font=dict(size=14)
                     )

    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    
#-----End of Page 3 (Widen ITZY's Listenership)-----#    
    

#-----Start of Page 4 (Spotlighting ITZY)-----#

elif my_page == 'Spotlighting ITZY':
    st.title("Spotlighting ITZY")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.image('images/spotlighting itzy.png',use_column_width=True)
    st.markdown("Another suggestion that we have for ITZY’s management team is to create a spotlight for ITZY. That means releasing songs on months with least activity from other K-Pop girl groups in order to focus the audience attention on ITZY.")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 25px;">Other most streamed K-Pop girl groups tend to release their songs on Spotify during April onwards</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    a1,a2 = st.beta_columns((4,1))
    a1.image('images/Timeline Top4.png',use_column_width=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("According to our data, the other most streamed K-Pop girl groups tend to release their songs on Spotify during April onwards (with the exception of MOMOLAND). Upon seeing this pattern we wanted to see how this  contrast that with ITZY’s past releases. ")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 25px;">ITZY’s previous release dates do not coincide with other K-Pop girl groups aside from the ‘Not Shy’ release</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    b1,b2 = st.beta_columns((4,1))
    b1.image('images/Timeline ITZY.png',use_column_width=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("We saw that ITZY released their debut album on February 2019 and another EP on March 2020, which are outside the April to December period mentioned earlier—however two of their releases were within that period.")
    

    
#-----End of Page 4 (Spotlighting ITZY)-----#


#-----Start of Page 5 (Recommendations)-----#

elif my_page == 'Recommendations':
    
    @st.cache
    def load_data(URL):
        data = pd.read_csv(URL)
        return data
    
    df6 = load_data("datasets/itzyfeatures_vs_reco.csv")
    df6.sort_values(by='value',ascending=True,inplace=True)
    
    st.title("Recommendations")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 25px;">Artists to collaborate with</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-size: 20px;">1. Based on song similarity</div>',unsafe_allow_html=True)  
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    i1,i2,i3,i4,i5 = st.beta_columns(5)
    i1.image('images/pop/Sigma.jpg',use_column_width=True)
    i1.markdown('<div style="text-align: center;color: #c6793a; font-weight: bold;">Sigma</div>',unsafe_allow_html=True)
    i1.markdown('<div style="text-align: center;color: #666666; font-style: italic">3.2×10^-3</div>',unsafe_allow_html=True)
    i2.image('images/rnb/Drake.jpg',use_column_width=True)
    i2.markdown('<div style="text-align: center;color: #77a9b4; font-weight: bold;">Drake</div>',unsafe_allow_html=True)
    i2.markdown('<div style="text-align: center;color: #666666; font-style: italic">86.39×10^-3</div>',unsafe_allow_html=True)
    i3.image('images/rap/Lost Kings.jpg',use_column_width=True)
    i3.markdown('<div style="text-align: center;color: #565b7b; font-weight: bold;">Lost Kings</div>',unsafe_allow_html=True)
    i3.markdown('<div style="text-align: center;color: #666666; font-style: italic">85.18×10^-3</div>',unsafe_allow_html=True)
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space # 
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-size: 20px;">2. Based on the number of streams of his/her charting songs</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    h1,h2,h3,h4,h5 = st.beta_columns(5)
    h1.image('images/pop/Camila.jpg',use_column_width=True)
    h1.markdown('<div style="text-align: center;color: #c6793a; font-weight: bold;">Camila Cabello</div>',unsafe_allow_html=True)
    h1.markdown('<div style="text-align: center;color: #666666; font-style: italic">96.9M Streams</div>',unsafe_allow_html=True)
    h1.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    h2.image('images/rnb/Drake.jpg',use_column_width=True)
    h2.markdown('<div style="text-align: center;color: #77a9b4; font-weight: bold;">Drake</div>',unsafe_allow_html=True)
    h2.markdown('<div style="text-align: center;color: #666666; font-style: italic">82.7M Streams</div>',unsafe_allow_html=True)
    h2.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    h3.image('images/rap/Charlie Puth.jpg',use_column_width=True)
    h3.markdown('<div style="text-align: center;color: #565b7b; font-weight: bold;">Charlie Puth</div>',unsafe_allow_html=True)
    h3.markdown('<div style="text-align: center;color: #666666; font-style: italic">69.8M Streams</div>',unsafe_allow_html=True)
    h3.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="font-weight: bold;font-size: 25px;">What musical direction to pursue</div>',unsafe_allow_html=True)
    st.markdown('<div style="color:gray;font-size: 18px;">ITZY can also base their collaboration decision on the genres\' notable audio features</div>',unsafe_allow_html=True)
    
    colors = ['#C6793A','#77A9B4', '#565B7B','#B88F89']

    fig = px.bar(df6, x="value", y="variable", color = "category", orientation='h', color_discrete_sequence=colors,
                 text=df6["value"].apply(lambda x: '{0:1.2f}'.format(x)))

    fig.update_traces(textposition='outside',
                  textfont=dict(size=14), width = 0.2)


    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode=False, barmode='group',
                  xaxis = {'title': 'Audio Features Value', 'range': [0, 1],
                           'showticklabels' : True, 'showgrid' : False,'zeroline': False,
                           'showline':True, 'linecolor':'#999999', 'tickfont':dict(color= '#999999'),
                           'titlefont' : dict(color = '#999999', size = 16), 'fixedrange':True},
                  yaxis = {'title': '', 'showgrid' : False,'zeroline': False, 'fixedrange':True},
                  margin=dict(l=0, r=0, b=0, t=25, pad=15), font=dict(size=14)
                 )

    config={'displayModeBar': False}

    st.plotly_chart(fig, use_container_width=True, config=config)
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('If ITZY wants to create a cheerful song, they can consider collaborating with someone under the R&B genre. If they want to create an energetic song, someone under the Pop genre might be a good fit. And if they want to create a danceable song, a collaboration with someone under the Rap genre is a fitting choice.',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="text-align: left; font-weight: bold;font-size: 25px;">Song release activities on months of January to March are few to none, desirable for future ITZY releases</div>',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    f1,f2 = st.beta_columns((4,1))
    f1.image('images/Timeline Empty.png',use_column_width=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('ITZY’s management team can consider doing future releases on the months of January to March, as data from previous years show that release activities from other most streamed K-Pop girl groups tend to be few to none.',unsafe_allow_html=True)
    
#-----End of Page 5 (Recommendation)-----#


#-----Start of Page 6 (The Team)-----#
elif my_page == 'The Team':
    
    st.title("The Team")
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    g1,g2,g3 = st.beta_columns(3)
    g1.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Mikee Jazmines</div>',unsafe_allow_html=True)
    g1.markdown('<div style="text-align: left;font-style: italic;color: gray;">Mentor</div>',unsafe_allow_html=True)
    g1.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    g1.markdown("""<a style='display: block; color:#77A9B4;text-decoration: none;' href="https://www.linkedin.com/in/mikee-jazmines-059b48bb/">LinkedIn</a>""",unsafe_allow_html=True)
    g1.markdown("""<a style='display: block; color:#77A9B4; text-decoration: none;' href="https://github.com/mikeejazmines"> GitHub </a>""", unsafe_allow_html=True)
    
    g2.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Andrei Labayan</div>',unsafe_allow_html=True)
    g2.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g2.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    g2.markdown("""<a style='display: block; color:#77A9B4;text-decoration: none;' href="https://www.linkedin.com/in/andrei-gabriel-labayan-48a8ba1a4/">LinkedIn</a>""",unsafe_allow_html=True)
    g2.markdown("""<a style='display: block; color:#77A9B4; text-decoration: none;' href="https://github.com/aalabayan"> GitHub </a>""", unsafe_allow_html=True)
    
    g3.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Patrick Nuguid</div>',unsafe_allow_html=True)
    g3.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g3.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    g3.markdown("""<a style='display: block; color:#77A9B4;text-decoration: none;' href="https://www.linkedin.com/in/patricknuguid/">LinkedIn</a>""",unsafe_allow_html=True)
    g3.markdown("""<a style='display: block; color:#77A9B4; text-decoration: none;' href="https://github.com/halubibi13"> GitHub </a>""", unsafe_allow_html=True)
    
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    
    g4,g5,g6 = st.beta_columns(3)

    g4.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Phoemela Ballaran</div>',unsafe_allow_html=True)
    g4.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g4.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    g4.markdown("""<a style='display: block; color:#77A9B4;text-decoration: none;' href="https://www.linkedin.com/in/phoemela-ballaran/">LinkedIn</a>""",unsafe_allow_html=True)
    g4.markdown("""<a style='display: block; color:#77A9B4; text-decoration: none;' href="https://github.com/phoemelaballaran"> GitHub </a>""", unsafe_allow_html=True)
    
    g5.markdown('<div style="text-align: left;font-size: large;font-weight: bold;">Razel Manapat</div>',unsafe_allow_html=True)
    g5.markdown('<div style="text-align: left;font-style: italic;color: gray;">Member</div>',unsafe_allow_html=True)
    g5.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    g5.markdown("""<a style='display: block; color:#77A9B4;text-decoration: none;' href="https://www.linkedin.com/in/razel-manapat-745650166/">LinkedIn</a>""",unsafe_allow_html=True)
    g5.markdown("""<a style='display: block; color:#77A9B4; text-decoration: none;' href="https://github.com/razelmanapat"> GitHub </a>""", unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    g7,g8 = st.beta_columns((1,3))
    g7.image('images/eskwelabs_logo_sprint2.png',use_column_width=True)
    st.markdown('*This is a sprint project of Eskwelabs Cohort 6 Data Science Fellows presented on 6 February 2021. Eskwelabs is an online data upskilling school in the Philippines. Their mission is to drive social mobility in the future of work through data skills education.*',unsafe_allow_html=True)
    st.markdown('<div style="color: transparent;">.</div>',unsafe_allow_html=True) # space #
    st.markdown("""<a style='display: block; color:#77A9B4; text-decoration: none;' href="https://www.eskwelabs.com"> Know more about Eskwelabs.</a>""", unsafe_allow_html=True)
    
#-----End of Page 6 (The Team)-----#
