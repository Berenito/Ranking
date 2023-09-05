# Ranking (Development of EUF ranking system)

Version: 2023-01-28

The goal of this code is to provide a structure to the implementation of the ranking algorithms, 
so they can be immediately applied to any available dataset. It also provides a possibility to gain 
insights about the games datasets and the way to export the results to reasonable formats.

## Folder structure

* `algos` - specific ranking-algorithm functions 
* `classes` - general concepts for implementing the algorithms
* `utils` - various utility functions
* `tests` - scripts validating the implemented algorithms

## Runnable Scripts

To initialize the code, do the following:
* Clone the repo using `git clone https://github.com/Berenito/Ranking.git`
* Create python env, install packages to it using `pip install requirements.txt`

All important constants and definitions are located in the `definitions.py` file.

### `prepare_data.py`

Take data from all the CSV files in the input folder and join them to create a big Game Table with the clean data;
export some preliminary summary statistics (no rankings are calculated here).

Prerequisites:
* Prepare a folder with the tournament result data - CSV files named `games_<suffix>.csv` with columns `Tournament, Date, Team_1, Team_2,
  Score_1, Score_2, Division`
* Add to the same folder a CSV file with columns `Team`, `Aliases` specifying the teams in the EUF system; multiple aliases can be
  defined for each team in the same row, separated with `, ` (filename should be `teams-<division>.txt`)
* Add to the same folder a CSV file with columns `Team`, `Tournament` specifying that the given team has met the
  EUF roster requirements for the particular tournament (filename should be `teams_at_tournaments-<division>.txt`)

Arguments:
* `--input <INPUT>` - path to the folder with all necessary files
* `--season <SEASON>` - current year
* `--output <OUTPUT>` - path to the folder to save the output CSVs
* `[--division <DIVISION>]` - women/mixed/open/all (default "all")

Procedure:
* If division is "all", repeat the next steps for all three divisions
* Read the tournament results CSVs and take only the games for the given division
* Read the list of EUF teams; replace aliases where applicable
* Read teams at tournaments list; add suffix to all teams without EUF-sanctioned roster for the given tournament
* Process the Game Table (check for invalid rows)
* Calculate basic statistics for the season (without any rankings)
* Save the output CSVs

Outputs:
* CSVs with Games, Tournaments, Calendar, and Summary (without any rankings)

### `calculate_rankings.py`

Calculate the rankings for all defined algorithms.

Prerequisites:
* Run `prepare_data.py` script (use its output path as input to this script)
* Specify the algorithms in `definitions.py`

Arguments:
* `--input <INPUT>`- path to the folder with all necessary files
* `--season <SEASON>` - current year
* `--date <DATE>` - date of calculation
* `--output <OUTPUT>` - path to save the output files
* `[--division <DIVISION>]` - women/mixed/open/all (default "all")

Outputs:
* CSV with Games, Summary (including ratings); pickle with GamesDataset object

### `export_to_datapane.py`

Export the Ranking data to the Datapane application (all divisions and all the algorithms together).

Prerequisites:
* Create the datapane account and get the token for app deployment
* Run `calculate_rankings.py` script for the same date as specified here and desired division (use its output path as input to this script)

Arguments:
* `--input <INPUT>`- path to the folder with all necessary files
* `--season <SEASON>` - current year
* `--token <TOKEN>` - datapane token for logging in
* `--date <DATE>` - date of calculation
* `[--division <DIVISION>]` - women/mixed/open/all (default "all")

Outputs:
* Datapane webpage with deployed application

*Note: Don't panic if the script finished with `requests.exceptions.HTTPError: 502 Server Error: Bad Gateway for url:`,
it probably succeeded and the new version is uploaded.*
  


