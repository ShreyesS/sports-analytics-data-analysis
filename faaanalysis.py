# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

first_season = "atp_matches_2018.csv"
second_season = "atp_matches_2019.csv"
third_season = "atp_matches_2020.csv"
fourth_season = "atp_matches_2021.csv"
fifth_season = "atp_matches_2022.csv"

full_df = pd.concat([pd.read_csv(first_season),pd.read_csv(second_season),pd.read_csv(third_season),pd.read_csv(fourth_season),pd.read_csv(fifth_season)])

full_df.shape

"""# 2018-2021 Seasons VS 2022 Season

2018, 2019, 2020, 2021 (just replace one)

## Stats Giver
"""

#function that takes in season and give you the values you want

def stats_giver(season):
  #win_df
  selected_columns_win = ["winner_name","w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved"]
  felix = "Felix Auger Aliassime"

  win_df = season.loc[season["winner_name"] == felix, selected_columns_win]
  win_df.rename(columns={'winner_name': 'name'}, inplace=True)
  win_df.rename(columns={'w_ace': '#aces'}, inplace=True)
  win_df.rename(columns={'w_svpt': '#svpts'}, inplace=True)
  win_df.rename(columns={'w_df': '#df'}, inplace=True)
  win_df.rename(columns={'w_2ndWon': '#2ndWon'}, inplace=True)
  win_df.rename(columns={'w_bpFaced': '#bpFaced'}, inplace=True)
  win_df.rename(columns={'w_bpSaved': '#bpSaved'}, inplace=True)

  #lose_df
  selected_columns_lose = ["loser_name", "l_ace", "l_svpt", "l_df", "l_2ndWon", "l_bpFaced", "l_bpSaved"]
  lose_df = season.loc[season["loser_name"] == felix, selected_columns_lose]
  lose_df.rename(columns={'loser_name': 'name'}, inplace=True)
  lose_df.rename(columns={'l_ace': '#aces'}, inplace=True)
  lose_df.rename(columns={'l_svpt': '#svpts'}, inplace=True)
  lose_df.rename(columns={'l_df': '#df'}, inplace=True)
  lose_df.rename(columns={'l_2ndWon': '#2ndWon'}, inplace=True)
  lose_df.rename(columns={'l_bpFaced': '#bpFaced'}, inplace=True)
  lose_df.rename(columns={'l_bpSaved': '#bpSaved'}, inplace=True)

  #final_df
  final_df = pd.concat([win_df, lose_df], ignore_index=True)

  final_df["%aces"] = (final_df["#aces"] / final_df["#svpts"])*100
  final_df["%bpSaved"] = (final_df["#bpSaved"] / final_df["#bpFaced"])*100

  #all of these are PER MATCH (especially the % ones)
  avg_num_aces = final_df['#aces'].mean()
  avg_percentage_aces_per_match = final_df['%aces'].mean()
  avg_num_df = final_df['#df'].mean()
  avg_num_2ndWon = final_df['#2ndWon'].mean()
  avg_num_bpFaced = final_df['#bpFaced'].mean()
  avg_num_bpSaved = final_df['#bpSaved'].mean()

  avg_percentage_bpSaved_per_match = final_df['%bpSaved'].mean()

  #displaying results
  print("Average #aces per match: " + str(avg_num_aces))
  print("Average % aces per match: " + str(avg_percentage_aces_per_match))
  print("Average #df per match: " + str(avg_num_df))
  print("Average #2nd serves won per match: " + str(avg_num_2ndWon))
  print("Average #bp faced per match: " + str(avg_num_bpFaced))
  print("Average #bp saved per match: " + str(avg_num_bpSaved))
  print("Average % bp saved per match: " + str(avg_percentage_bpSaved_per_match))

stats_giver(pd.read_csv(first_season))

stats_giver(pd.read_csv(fifth_season))

"""✅"data revealed that %bp saved per match ..."

## Grouped Bar Chart
"""

#group bar char plotter of 2 seasons

def grouped_bar_plotter(season1, season2):

  def stats_getter(season):
    #win_df
    selected_columns_win = ["winner_name","w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
    felix = "Felix Auger Aliassime"

    win_df = season.loc[season["winner_name"] == felix, selected_columns_win]
    win_df.rename(columns={'winner_name': 'name'}, inplace=True)
    win_df.rename(columns={'w_ace': '#aces'}, inplace=True)
    win_df.rename(columns={'w_svpt': '#svpts'}, inplace=True)
    win_df.rename(columns={'w_df': '#df'}, inplace=True)
    win_df.rename(columns={'w_2ndWon': '#2ndWon'}, inplace=True)
    win_df.rename(columns={'w_bpFaced': '#bpFaced'}, inplace=True)
    win_df.rename(columns={'w_bpSaved': '#bpSaved'}, inplace=True)
    win_df.rename(columns={'l_bpFaced': '#bpOppFaced'}, inplace=True)
    win_df.rename(columns={'l_bpSaved': '#bpOppSaved'}, inplace=True)

    #lose_df
    selected_columns_lose = ["loser_name", "l_ace", "l_svpt", "l_df", "l_2ndWon", "l_bpFaced", "l_bpSaved", "w_bpFaced", "w_bpSaved"]
    lose_df = season.loc[season["loser_name"] == felix, selected_columns_lose]
    lose_df.rename(columns={'loser_name': 'name'}, inplace=True)
    lose_df.rename(columns={'l_ace': '#aces'}, inplace=True)
    lose_df.rename(columns={'l_svpt': '#svpts'}, inplace=True)
    lose_df.rename(columns={'l_df': '#df'}, inplace=True)
    lose_df.rename(columns={'l_2ndWon': '#2ndWon'}, inplace=True)
    lose_df.rename(columns={'l_bpFaced': '#bpFaced'}, inplace=True)
    lose_df.rename(columns={'l_bpSaved': '#bpSaved'}, inplace=True)
    win_df.rename(columns={'w_bpFaced': '#bpOppFaced'}, inplace=True)
    win_df.rename(columns={'w_bpSaved': '#bpOppSaved'}, inplace=True)

    #final_df
    final_df = pd.concat([win_df, lose_df], ignore_index=True)

    #final_df["%aces"] = (final_df["#aces"] / final_df["#svpts"])*100
    #final_df["%df"] = (final_df["#df"] / final_df["#svpts"])*100
    final_df["%bpSaved"] = (final_df["#bpSaved"] / final_df["#bpFaced"])*100

    final_df["#bpConverted"] = final_df["#bpOppFaced"] - final_df["#bpOppSaved"]

    #all of these are PER MATCH (especially the % ones)
    avg_num_aces = final_df['#aces'].mean()
    #avg_percentage_aces_per_match = final_df['%aces'].mean()
    avg_num_df = final_df['#df'].mean()
    avg_num_2ndWon = final_df['#2ndWon'].mean()
    avg_num_bpFaced = final_df['#bpFaced'].mean()
    avg_num_bpSaved = final_df['#bpSaved'].mean()
    avg_percentage_bpSaved_per_match = final_df['%bpSaved'].mean()

    avg_bp_converted = final_df["#bpConverted"].mean()

    values = [avg_num_aces, avg_num_df, avg_num_2ndWon,
            avg_num_bpFaced, avg_bp_converted]

    return values

  # Data for the two intervals
  factors = ['#aces', '#df', '#2nd W', '#bp F', '#bp C']
  interval1_values = stats_getter(season1)
  interval2_values = stats_getter(season2)

  # Create an array for the x-axis positions
  x = np.arange(len(factors))

  # Plot the grouped bar graph
  width = 0.35  # Width of the bars
  #light red - #ff726f; light blue - #00c3e3
  plt.bar(x - width/2, interval1_values, width, color = '#ff726f', label='First Season (2018)')
  for i, value in enumerate(interval1_values):
    plt.text(i, value, str(round(value, 2)), ha='right', va='bottom')

  plt.bar(x + width/2, interval2_values, width, color = '#00c3e3', label='Fifth (most recent) Season (2022)')
  for i, value in enumerate(interval2_values):
    plt.text(i, value, str(round(value, 2)), ha='left', va='bottom')



  # Set the axis labels, title, and legend
  plt.xlabel('Factors')
  plt.ylabel('Average Stat Values (per match)')
  plt.title('Change in FAA Factors from 2018 to 2022')
  plt.xticks(x, factors)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.ylim(0, 18)

  # Display the plot
  plt.show()

grouped_bar_plotter(pd.read_csv(first_season), pd.read_csv(fifth_season))

#%bp S missing cuz it is way bigger than the other values;
#i can just talk about how that changed too (and maybe add why I did not include it in the graph above)

# % group bar char plotter

def grouped_bar_percentage_plotter(season1, season2):

  def stats_getter(season):
    #win_df
    selected_columns_win = ["winner_name","w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
    felix = "Felix Auger Aliassime"

    win_df = season.loc[season["winner_name"] == felix, selected_columns_win]
    win_df.rename(columns={'winner_name': 'name'}, inplace=True)
    win_df.rename(columns={'w_ace': '#aces'}, inplace=True)
    win_df.rename(columns={'w_svpt': '#svpts'}, inplace=True)
    win_df.rename(columns={'w_df': '#df'}, inplace=True)
    win_df.rename(columns={'w_2ndWon': '#2ndWon'}, inplace=True)
    win_df.rename(columns={'w_bpFaced': '#bpFaced'}, inplace=True)
    win_df.rename(columns={'w_bpSaved': '#bpSaved'}, inplace=True)
    win_df.rename(columns={'l_bpFaced': '#bpOppFaced'}, inplace=True)
    win_df.rename(columns={'l_bpSaved': '#bpOppSaved'}, inplace=True)

    #lose_df
    selected_columns_lose = ["loser_name", "l_ace", "l_svpt", "l_df", "l_2ndWon", "l_bpFaced", "l_bpSaved", "w_bpFaced", "w_bpSaved"]
    lose_df = season.loc[season["loser_name"] == felix, selected_columns_lose]
    lose_df.rename(columns={'loser_name': 'name'}, inplace=True)
    lose_df.rename(columns={'l_ace': '#aces'}, inplace=True)
    lose_df.rename(columns={'l_svpt': '#svpts'}, inplace=True)
    lose_df.rename(columns={'l_df': '#df'}, inplace=True)
    lose_df.rename(columns={'l_2ndWon': '#2ndWon'}, inplace=True)
    lose_df.rename(columns={'l_bpFaced': '#bpFaced'}, inplace=True)
    lose_df.rename(columns={'l_bpSaved': '#bpSaved'}, inplace=True)
    win_df.rename(columns={'w_bpFaced': '#bpOppFaced'}, inplace=True)
    win_df.rename(columns={'w_bpSaved': '#bpOppSaved'}, inplace=True)

    #final_df
    final_df = pd.concat([win_df, lose_df], ignore_index=True)

    final_df["%aces"] = (final_df["#aces"] / final_df["#svpts"])*100
    final_df["%df"] = (final_df["#df"] / final_df["#svpts"])*100
    final_df["%bpSaved"] = (final_df["#bpSaved"] / final_df["#bpFaced"])*100

    final_df["#bpConverted"] = final_df["#bpOppFaced"] - final_df["#bpOppSaved"] #this gives FAA's BP converted
    final_df["%bpConverted"] = (final_df["#bpConverted"] / final_df["#bpOppFaced"])*100

    #all of these are PER MATCH (especially the % ones)
    avg_percentage_aces_per_match = final_df['%aces'].mean()
    avg_percentage_df_per_match = final_df['%df'].mean()
    avg_percentage_bpSaved_per_match = final_df['%bpSaved'].mean()
    avg_percentage_bpConverted_per_match = final_df['%bpConverted'].mean()

    values = [avg_percentage_aces_per_match, avg_percentage_df_per_match, avg_percentage_bpConverted_per_match,
              avg_percentage_bpSaved_per_match]

    return values

  # Data for the two intervals
  factors = ['%aces', '%df', '%bp C', '%bp S']
  interval1_values = stats_getter(season1)
  interval2_values = stats_getter(season2)

  # Create an array for the x-axis positions
  x = np.arange(len(factors))

  # Plot the grouped bar graph
  width = 0.35  # Width of the bars
  #light red - #ff726f; light blue - #00c3e3
  plt.bar(x - width/2, interval1_values, width, color = '#ff726f', label='First Season (2018)')
  for i, value in enumerate(interval1_values):
    plt.text(i, value, str(round(value, 2)), ha='right', va='bottom')
  plt.bar(x + width/2, interval2_values, width, color = '#00c3e3', label='Fifth (most recent) Season (2022)')
  for i, value in enumerate(interval2_values):
    plt.text(i, value, str(round(value, 2)), ha='left', va='bottom')


  # Set the axis labels, title, and legend
  plt.xlabel('Factors')
  plt.ylabel('Average Stat Values (per match)')
  plt.title('Percentage Change in FAA Factors from 2018 to 2022')
  plt.xticks(x, factors)
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.ylim(0, 70)

  # Display the plot
  plt.show()

grouped_bar_percentage_plotter(pd.read_csv(first_season), pd.read_csv(fifth_season))

stats_giver(pd.read_csv(first_season))

stats_giver(pd.read_csv(fifth_season))

"""✅tennis is a game of small margins so these change in numbers is huge! FAA has improved over the years. 2022 was his best year, in terms of these determinants ##

# Finals Lost(8) VS Finals Won(4)

## WIN
"""

#win_df
season = pd.read_csv(fifth_season)

selected_columns_win = ["tourney_name", "winner_name", "loser_name", "round", "winner_rank", "loser_rank", "w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
felix = "Felix Auger Aliassime"
final = "F"

win_df = season.loc[season["winner_name"] == felix, selected_columns_win]
win_df = win_df.loc[win_df["round"] == final, selected_columns_win]

#win_df = win_df.drop(win_df[win_df['tourney_name'] == 'Atp Cup'].index) #getting rid of when he won ATP Cup for Canada
win_df.head(30)

#pass in a season => gives you the stats for when FAA WON a final

def final_winner_stats_giver(season):
  #win_df
  selected_columns_win = ["winner_name","w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
  felix = "Felix Auger Aliassime"

  win_df = season.loc[season["winner_name"] == felix, selected_columns_win]
  win_df.rename(columns={'winner_name': 'name'}, inplace=True)
  win_df.rename(columns={'w_ace': '#aces'}, inplace=True)
  win_df.rename(columns={'w_svpt': '#svpts'}, inplace=True)
  win_df.rename(columns={'w_df': '#df'}, inplace=True)
  win_df.rename(columns={'w_2ndWon': '#2ndWon'}, inplace=True)
  win_df.rename(columns={'w_bpFaced': '#bpFaced'}, inplace=True)
  win_df.rename(columns={'w_bpSaved': '#bpSaved'}, inplace=True)
  win_df.rename(columns={'l_bpFaced': '#bpOppFaced'}, inplace=True)
  win_df.rename(columns={'l_bpSaved': '#bpOppSaved'}, inplace=True)



  #final_df
  final_df = win_df

  final_df["%aces"] = (final_df["#aces"] / final_df["#svpts"])*100
  final_df["%df"] = (final_df["#df"] / final_df["#svpts"])*100
  final_df["%bpSaved"] = (final_df["#bpSaved"] / final_df["#bpFaced"])*100

  final_df["#bpConverted"] = final_df["#bpOppFaced"] - final_df["#bpOppSaved"]
  final_df["%bpConverted"] = (final_df["#bpConverted"] / final_df["#bpOppFaced"])*100


  #all of these are PER MATCH (especially the % ones)
  avg_num_aces = final_df['#aces'].mean()
  avg_percentage_aces_per_match = final_df['%aces'].mean()
  avg_num_df = final_df['#df'].mean()
  avg_percentage_df = final_df['%df'].mean()
  avg_num_2ndWon = final_df['#2ndWon'].mean()
  avg_num_bpFaced = final_df['#bpFaced'].mean()
  avg_num_bpSaved = final_df['#bpSaved'].mean()
  avg_percentage_bpSaved_per_match = final_df['%bpSaved'].mean()
  avg_bp_converted = final_df["#bpConverted"].mean()
  avg_percentage_bpConverted_per_match = final_df['%bpConverted'].mean()

  '''values = [avg_num_aces, avg_percentage_aces_per_match, avg_num_df, avg_percentage_df, avg_num_2ndWon,
            avg_num_bpFaced, avg_num_bpSaved, avg_percentage_bpSaved_per_match, avg_bp_converted,
            avg_percentage_bpConverted_per_match]
            '''
  values = [avg_num_aces, avg_num_df, avg_num_2ndWon, avg_num_bpFaced, avg_num_bpSaved, avg_bp_converted]

  print("Average % aces per match: " + str(avg_percentage_aces_per_match))
  print("Average % df per match: " + str(avg_percentage_df))
  print("Average % bp converted per match: " + str(avg_percentage_bpConverted_per_match))
  print("Average % bp saved per match: " + str(avg_percentage_bpSaved_per_match))

  return values

"""## LOSE"""

#lose_df
season = full_df

selected_columns_lose = ["tourney_name", "winner_name", "loser_name", "round", "winner_rank", "loser_rank", "l_ace", "l_svpt", "l_df", "l_2ndWon", "l_bpFaced", "l_bpSaved", "w_bpFaced", "w_bpSaved"]
felix = "Felix Auger Aliassime"
final = "F"

lose_df = season.loc[season["loser_name"] == felix, selected_columns_lose]
lose_df = lose_df.loc[lose_df["round"] == final, selected_columns_lose]

lose_df = lose_df.drop(lose_df[lose_df['winner_name'] == 'Andrey Rublev'].index) #just makeing it 8 finals
lose_df.head(30)

#pass in a season => gives you the stats for when FAA LOST a final

def final_loser_stats_giver(season):
  #lose_df
  selected_columns_lose = ["loser_name", "l_ace", "l_svpt", "l_df", "l_2ndWon", "l_bpFaced", "l_bpSaved", "w_bpFaced", "w_bpSaved"]
  lose_df = season.loc[season["loser_name"] == felix, selected_columns_lose]
  lose_df.rename(columns={'loser_name': 'name'}, inplace=True)
  lose_df.rename(columns={'l_ace': '#aces'}, inplace=True)
  lose_df.rename(columns={'l_svpt': '#svpts'}, inplace=True)
  lose_df.rename(columns={'l_df': '#df'}, inplace=True)
  lose_df.rename(columns={'l_2ndWon': '#2ndWon'}, inplace=True)
  lose_df.rename(columns={'l_bpFaced': '#bpFaced'}, inplace=True)
  lose_df.rename(columns={'l_bpSaved': '#bpSaved'}, inplace=True)
  lose_df.rename(columns={'w_bpFaced': '#bpOppFaced'}, inplace=True)
  lose_df.rename(columns={'w_bpSaved': '#bpOppSaved'}, inplace=True)

  #final_df
  final_df = lose_df

  final_df["%aces"] = (final_df["#aces"] / final_df["#svpts"])*100
  final_df["%df"] = (final_df["#df"] / final_df["#svpts"])*100
  final_df["%bpSaved"] = (final_df["#bpSaved"] / final_df["#bpFaced"])*100

  final_df["#bpConverted"] = final_df["#bpOppFaced"] - final_df["#bpOppSaved"]
  final_df["%bpConverted"] = (final_df["#bpConverted"] / final_df["#bpOppFaced"])*100


  #all of these are PER MATCH (especially the % ones)
  avg_num_aces = final_df['#aces'].mean()
  avg_percentage_aces_per_match = final_df['%aces'].mean()
  avg_num_df = final_df['#df'].mean()
  avg_percentage_df = final_df['%df'].mean()
  avg_num_2ndWon = final_df['#2ndWon'].mean()
  avg_num_bpFaced = final_df['#bpFaced'].mean()
  avg_num_bpSaved = final_df['#bpSaved'].mean()
  avg_percentage_bpSaved_per_match = final_df['%bpSaved'].mean()
  avg_bp_converted = final_df["#bpConverted"].mean()
  avg_percentage_bpConverted_per_match = final_df['%bpConverted'].mean()

  values = [avg_num_aces, avg_num_df, avg_num_2ndWon, avg_num_bpFaced, avg_num_bpSaved, avg_bp_converted]

  '''values = [avg_num_aces, avg_percentage_aces_per_match, avg_num_df, avg_percentage_df, avg_num_2ndWon,
            avg_num_bpFaced, avg_num_bpSaved, avg_percentage_bpSaved_per_match, avg_bp_converted,
            avg_percentage_bpConverted_per_match]
  '''
  print("Average % aces per match: " + str(avg_percentage_aces_per_match))
  print("Average % df per match: " + str(avg_percentage_df))
  print("Average % bp converted per match: " + str(avg_percentage_bpConverted_per_match))
  print("Average % bp saved per match: " + str(avg_percentage_bpSaved_per_match))

  return values

final_winner_stats_giver(win_df)

final_loser_stats_giver(lose_df)

"""✅picture below. you have the NUMBER radar plot and talk about percentages cuz those are **more** important

✅The times that he lost, sometimes he was ranked above, sometimes he was ranked below (third season), so look at determinants instead
"""

final_winner_stats_giver(win_df)

final_loser_stats_giver(lose_df)

"""## Radar Plotting"""

import numpy as np
import matplotlib.pyplot as plt

# Data for the radar chart
#factors = ['#aces', '%aces', '#df', '%df', '#2nd W', '#bp F', '#bp S', '%bp S', '#bp C', '%bp C']
factors = ['#aces', '#df', '#2nd W', '#bp F', '#bp S','#bp C']
interval1_values = final_winner_stats_giver(win_df)
interval2_values = final_loser_stats_giver(lose_df)

# Make sure the length of interval values matches the number of factors
assert len(interval1_values) == len(factors), "Length mismatch between factors and interval1_values"
assert len(interval2_values) == len(factors), "Length mismatch between factors and interval2_values"

# Create an array for the angles
angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
angles += angles[:1]  # Close the circle

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

# Plot the data for the first interval
ax.plot(angles, interval1_values + [interval1_values[0]], linewidth=1, linestyle='solid', label='4 Finals Won')
ax.fill(angles, interval1_values + [interval1_values[0]], alpha=0.25)

# Plot the data for the second interval
ax.plot(angles, interval2_values + [interval2_values[0]], linewidth=1, linestyle='solid', label='8 Finals Lost')
ax.fill(angles, interval2_values + [interval2_values[0]], alpha=0.25)

# Set the labels for each factor
ax.set_xticks(angles[:-1])
ax.set_xticklabels(factors)

# Set the radial axis labels and limits
ax.set_rlabel_position(0)
ax.set_ylim(0, 12)

# Add a legend and title
ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
ax.set_title('Comparing 8 Finals Lost VS 4 Finals Won', pad = 30)

ax.scatter(angles[:-1], interval1_values, color='black', s=8, label='Interval 1')
ax.scatter(angles[:-1], interval2_values, color='black', s=8, label='Interval 2')


#ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])  # Adjust the tick positions as needed
ax.tick_params(axis='y', labelsize=7)
ax.tick_params(axis='x', pad=20)

# Display the radar chart
plt.show()

"""# Triumphant Trio

Average Performance VS Triumphant Trio performance
"""

avg_performance_df = pd.concat([pd.read_csv(first_season),pd.read_csv(second_season),pd.read_csv(third_season),pd.read_csv(fourth_season)])

"""## Stats getter"""

def stats_getter(season):
  #win_df
  selected_columns_win = ["winner_name","w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
  felix = "Felix Auger Aliassime"

  win_df = season.loc[season["winner_name"] == felix, selected_columns_win]
  win_df.rename(columns={'winner_name': 'name'}, inplace=True)
  win_df.rename(columns={'w_ace': '#aces'}, inplace=True)
  win_df.rename(columns={'w_svpt': '#svpts'}, inplace=True)
  win_df.rename(columns={'w_df': '#df'}, inplace=True)
  win_df.rename(columns={'w_2ndWon': '#2ndWon'}, inplace=True)
  win_df.rename(columns={'w_bpFaced': '#bpFaced'}, inplace=True)
  win_df.rename(columns={'w_bpSaved': '#bpSaved'}, inplace=True)
  win_df.rename(columns={'l_bpFaced': '#bpOppFaced'}, inplace=True)
  win_df.rename(columns={'l_bpSaved': '#bpOppSaved'}, inplace=True)

  #lose_df
  selected_columns_lose = ["loser_name", "l_ace", "l_svpt", "l_df", "l_2ndWon", "l_bpFaced", "l_bpSaved", "w_bpFaced", "w_bpSaved"]
  lose_df = season.loc[season["loser_name"] == felix, selected_columns_lose]
  lose_df.rename(columns={'loser_name': 'name'}, inplace=True)
  lose_df.rename(columns={'l_ace': '#aces'}, inplace=True)
  lose_df.rename(columns={'l_svpt': '#svpts'}, inplace=True)
  lose_df.rename(columns={'l_df': '#df'}, inplace=True)
  lose_df.rename(columns={'l_2ndWon': '#2ndWon'}, inplace=True)
  lose_df.rename(columns={'l_bpFaced': '#bpFaced'}, inplace=True)
  lose_df.rename(columns={'l_bpSaved': '#bpSaved'}, inplace=True)
  lose_df.rename(columns={'w_bpFaced': '#bpOppFaced'}, inplace=True)
  lose_df.rename(columns={'w_bpSaved': '#bpOppSaved'}, inplace=True)

  #final_df
  final_df = pd.concat([win_df, lose_df], ignore_index=True)

  final_df["%aces"] = (final_df["#aces"] / final_df["#svpts"])*100
  final_df["%df"] = (final_df["#df"] / final_df["#svpts"])*100
  final_df["%bpSaved"] = (final_df["#bpSaved"] / final_df["#bpFaced"])*100
  final_df["#bpConverted"] = final_df["#bpOppFaced"] - final_df["#bpOppSaved"]
  final_df["%bpConverted"] = (final_df["#bpConverted"] / final_df["#bpOppFaced"])*100

  avg_num_aces = final_df['#aces'].mean()
  avg_percentage_aces_per_match = final_df['%aces'].mean()
  avg_num_df = final_df['#df'].mean()
  avg_percentage_df = final_df['%df'].mean()
  avg_num_2ndWon = final_df['#2ndWon'].mean()
  avg_num_bpFaced = final_df['#bpFaced'].mean()
  avg_num_bpSaved = final_df['#bpSaved'].mean()
  avg_percentage_bpSaved_per_match = final_df['%bpSaved'].mean()
  avg_bp_converted = final_df["#bpConverted"].mean()
  avg_percentage_bpConverted_per_match = final_df['%bpConverted'].mean()

  values = [avg_percentage_aces_per_match, avg_percentage_df, avg_percentage_bpConverted_per_match, avg_percentage_bpSaved_per_match]

  return values

"""## Triumphant Trio"""

#triumphant_trio
season = pd.read_csv(fifth_season)

selected_columns_win = ["tourney_name", "winner_name", "loser_name", "round", "winner_rank", "loser_rank", "w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved", 'l_bpFaced', 'l_bpSaved']
felix = "Felix Auger Aliassime"
final = "F"

tt_df = season.loc[season["winner_name"] == felix, selected_columns_win]
tt_df = tt_df.loc[tt_df["round"] == final, selected_columns_win]

#getting rid of ATP Cup and Rotterdam
tt_df = tt_df.drop(tt_df[tt_df['tourney_name'] == 'Atp Cup'].index)
tt_df = tt_df.drop(tt_df[tt_df['tourney_name'] == 'Rotterdam'].index) #getting rid of when he won ATP Cup for Canada
tt_df.head(30)

avg_num_aces = tt_df['w_ace'].mean()
avg_num_aces

def triumph_trio_func(season):
  #win_df
  selected_columns_win = ["winner_name","w_ace", "w_svpt", "w_df", "w_2ndWon", "w_bpFaced", "w_bpSaved", "l_bpFaced", "l_bpSaved"]
  felix = "Felix Auger Aliassime"

  win_df = season.loc[season["winner_name"] == felix, selected_columns_win]
  win_df.rename(columns={'winner_name': 'name'}, inplace=True)
  win_df.rename(columns={'w_ace': '#aces'}, inplace=True)
  win_df.rename(columns={'w_svpt': '#svpts'}, inplace=True)
  win_df.rename(columns={'w_df': '#df'}, inplace=True)
  win_df.rename(columns={'w_2ndWon': '#2ndWon'}, inplace=True)
  win_df.rename(columns={'w_bpFaced': '#bpFaced'}, inplace=True)
  win_df.rename(columns={'w_bpSaved': '#bpSaved'}, inplace=True)
  win_df.rename(columns={'l_bpFaced': '#bpOppFaced'}, inplace=True)
  win_df.rename(columns={'l_bpSaved': '#bpOppSaved'}, inplace=True)

  #final_df
  final_df = win_df

  final_df["%aces"] = (final_df["#aces"] / final_df["#svpts"])*100
  final_df["%df"] = (final_df["#df"] / final_df["#svpts"])*100
  final_df["%bpSaved"] = (final_df["#bpSaved"] / final_df["#bpFaced"])*100

  final_df["#bpConverted"] = final_df["#bpOppFaced"] - final_df["#bpOppSaved"]
  final_df["%bpConverted"] = (final_df["#bpConverted"] / final_df["#bpOppFaced"])*100


  #all of these are PER MATCH (especially the % ones)
  avg_num_aces = final_df['#aces'].mean()
  avg_percentage_aces_per_match = final_df['%aces'].mean()
  avg_num_df = final_df['#df'].mean()
  avg_percentage_df = final_df['%df'].mean()
  avg_num_2ndWon = final_df['#2ndWon'].mean()
  avg_num_bpFaced = final_df['#bpFaced'].mean()
  avg_num_bpSaved = final_df['#bpSaved'].mean()
  avg_percentage_bpSaved_per_match = final_df['%bpSaved'].mean()
  avg_bp_converted = final_df["#bpConverted"].mean()
  avg_percentage_bpConverted_per_match = final_df['%bpConverted'].mean()

  values = [avg_percentage_aces_per_match, avg_percentage_df, avg_percentage_bpConverted_per_match, avg_percentage_bpSaved_per_match]

  return values

stats_getter(avg_performance_df)

triumph_trio_func(tt_df)

factors = ['%aces', '%df', '%bp C', '%bp S']
interval1_values = stats_getter(avg_performance_df)
interval2_values = triumph_trio_func(tt_df)

# Create an array for the x-axis positions
x = np.arange(len(factors))

# Plot the grouped bar graph
width = 0.35  # Width of the bars
#light red - #ff726f; light blue - #00c3e3
plt.bar(x - width/2, interval1_values, width, label='Average Performance')
for i, value in enumerate(interval1_values):
  plt.text(i, value, str(round(value, 2)), ha='right', va='bottom')
plt.bar(x + width/2, interval2_values, width, label='Triumphant Trio')
for i, value in enumerate(interval2_values):
  plt.text(i, value, str(round(value, 2)), ha='left', va='bottom')


# Set the axis labels, title, and legend
plt.xlabel('Factors')
plt.ylabel('Average Stat Values (per match)')
plt.title('Triumphant Trio Performance VS Average Performance')
plt.xticks(x, factors)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0, 70)

# Display the plot
plt.show()

"""## Basel Masterclass"""

basel = tt_df
basel = basel.drop(basel[basel['tourney_name'] == 'Antwerp'].index)
basel = basel.drop(basel[basel['tourney_name'] == 'Florence'].index)

basel

factors = ['%aces', '%df', '%bp C', '%bp S']
interval1_values = stats_getter(avg_performance_df)
interval2_values = triumph_trio_func(tt_df)
interval3_values = triumph_trio_func(basel)

# Define the bar positions and widths
bar_width = 0.3
x = range(len(factors))

# Plot the bars for each group
plt.bar(x, interval1_values, width=bar_width, label='Average Performance')
plt.bar([i + bar_width for i in x], interval2_values, width=bar_width, label='Triumphant Trio')
plt.bar([i + 2 * bar_width for i in x], interval3_values, width=bar_width, label='Basel')

# Add value labels to the bars
for i, value in enumerate(interval1_values):
    plt.text(i, value + 0.5, round(value, 2), ha='center', fontsize = 7)
for i, value in enumerate(interval2_values):
    plt.text(i + bar_width, value + 0.5, round(value, 2), ha='center', fontsize = 7)
for i, value in enumerate(interval3_values):
    plt.text(i + 2 * bar_width, value + 0.5, round(value, 2), ha='center', fontsize = 7)


# Set plot labels and titles
plt.xlabel('Factors')
plt.ylabel('Average Stat Values (per match)')
plt.title('Basel Masterclass VS Triumphant Trio & Average Performance')

# Set the x-axis ticks and labels
plt.xticks([i + 1.5 * bar_width for i in x], factors)

# Add a legend
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Display the chart
plt.show()

final_winner_stats_giver(tt_df)

final_winner_stats_giver(basel)

"""Interesting to note that a higher number of 2nd serves won isn't necessarily a optimistic result. It is usually seen with a lower number of aces and ace percentage on the same line. This means that the lack of aces or also first serves being made results in more second serves being served by FAA. Therefore, making it more likely to see a higher number of second serves won (since more second serves are served). This trend seems to be backed on the other line graphs too: at Florence, FAA served his best % of aces, and had the least number of 2nd serves won"""
