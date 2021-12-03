# Ranking (Development of EUF ranking system)
Version: 2021-12-03

The goal of this code is to provide a structure to the implementation of the ranking algorithms so they can be immediately applied to any available dataset. It also provides a possibility to gain insights about the games datasets and the way to export the results to reasonable formats.

#### Requirements
Code is in Python, below are the packages outside of the standard library, the version in the parentheses is the one I was testing it on, but there should not be any version-dependency in the code.
* numpy (1.19.2)
* pandas (1.2.0)
* plotly (4.14.3) - if you want to export figures

#### Dataset format
Games dataset should be saved in a csv file with the following 6 columns:
* Tournament - name of the event
* Date - game date in YYYY-MM-DD format
* Team_1, Team_2 - participating teams
* Score_1, Score_2 - resulting scores

#### Folders
* algos - specific ranking algorithms scripts 
* data - csv files with the games datasets
* figures - place for exported figures
* reports - place for exported excel files

#### Scripts
* ranking_classes.py - classes to work with datasets and ranking algorithms
* helpfunctions_dataset.py - helpfunctions to analyse games datasets
* helpfunctions_excel.py - helpfunctions to ease the exporting to excel
* helpfunctions_plotly.py - helpfunctions to ease the making of plotly figures
* example_ranking_script.py - example of possible workflow

#### Algos
* example_algo.py - just an example to show how it works
* usau_algo.py - USAU ranking algorithm, functional version, but not 100% consistent with USAU rankings yet 

#### How to use it
* check example_ranking_script.py for an example of calculating the ratings for the given dataset
* some functions in helpfunctions_dataset.py can be also useful in the writing of particular ranking algos
* use help() to see the documentation, e.g. import ranking_classes; help(ranking_classes)

#### Next steps
* Fix possible issues in the code 
* Add more functionality
* Prepare more datasets for testing
* Prepare more algos for testing
* Make GUI for easier analysis ?
* ...

