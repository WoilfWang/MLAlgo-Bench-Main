You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named PUBG_Finish_Placement_Prediction_(Kernels_Only)_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
So, where we droppin' boys and girls?
Battle Royale-style video games have taken the world by storm. 100 players are dropped onto an island empty-handed and must explore, scavenge, and eliminate other players until only one is left standing, all while the play zone continues to shrink. 

PlayerUnknown's BattleGrounds (PUBG) has enjoyed massive popularity. With over 50 million copies sold, it's the fifth best selling game of all time, and has millions of active monthly players.  

The team at PUBG has made official game data available for the public to explore and scavenge outside of "The Blue Circle." This competition is not an official or affiliated PUBG site - Kaggle collected data made possible through the PUBG Developer API.

You are given over 65,000 games' worth of anonymized player data, split into training and testing sets, and asked to predict final placement from final in-game stats and initial player ratings. 
What's the best strategy to win in PUBG? Should you sit in one spot and hide your way into victory, or do you need to be the top shot? Let's let the data do the talking!

##  Evaluation Metric:
Submissions are evaluated on Mean Absolute Error between your predicted winPlacePerc and the observed winPlacePerc.

#### Submission File
For each Id in the test set, you must predict their placement as a percentage (0 for last, 1 for first place) for the winPlacePerc variable. The file should contain a header and have the following format:

    Id,winPlacePerc
    47734,0
    47735,0.5
    47736,0
    47737,1
    etc.


##  Dataset Description:
In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves. 
You are provided with a large number of anonymized PUBG game stats, formatted so that each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 player per group.
You must create a model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place). 

#### File descriptions

    train_V2.csv - the training set
    test_V2.csv - the test set
    sample_submission_V2.csv - a sample submission file in the correct format

#### Data fields

    DBNOs -  Number of enemy players knocked.
    assists -  Number of enemy players this player damaged that were killed by teammates.
    boosts -  Number of boost items used.
    damageDealt -  Total damage dealt. Note: Self inflicted damage is subtracted.
    headshotKills - Number of enemy players killed with headshots.
    heals - Number of healing items used.
    Id - Player’s Id
    killPlace -  Ranking in match of number of enemy players killed.
    killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”. 
    killStreaks - Max number of enemy players killed in a short amount of time.
    kills - Number of enemy players killed.
    longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
    matchDuration - Duration of match in seconds.
    matchId - ID to identify match. There are no matches that are in both the training and testing set.
    matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches. 
    rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
    revives - Number of times this player revived teammates.
    rideDistance - Total distance traveled in vehicles measured in meters.
    roadKills - Number of kills while in a vehicle.
    swimDistance - Total distance traveled by swimming measured in meters.
    teamKills - Number of times this player killed a teammate.
    vehicleDestroys - Number of vehicles destroyed.
    walkDistance - Total distance traveled on foot measured in meters.
    weaponsAcquired - Number of weapons picked up.
    winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”. 
    groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
    numGroups - Number of groups we have data for in the match.
    maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements. 
    winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

test_V2.csv - column name: Id, groupId, matchId, assists, boosts, damageDealt, DBNOs, headshotKills, heals, killPlace, killPoints, kills, killStreaks, longestKill, matchDuration, matchType, maxPlace, numGroups, rankPoints, revives, rideDistance, roadKills, swimDistance, teamKills, vehicleDestroys, walkDistance, weaponsAcquired, winPoints
train_V2.csv - column name: Id, groupId, matchId, assists, boosts, damageDealt, DBNOs, headshotKills, heals, killPlace, killPoints, kills, killStreaks, longestKill, matchDuration, matchType, maxPlace, numGroups, rankPoints, revives, rideDistance, roadKills, swimDistance, teamKills, vehicleDestroys, walkDistance, weaponsAcquired, winPoints, winPlacePerc


## Dataset folder Location: 
../../kaggle-data/pubg-finish-placement-prediction. In this folder, there are the following files you can use: test_V2.csv, sample_submission_V2.csv, train_V2.csv

## Solution Description:
There were three key steps necessary in our solution.
The first key step, and most important, was understanding the importance of killPlace, which contains some leakage of the target variable.  This knowledge was recognised publicly, but was not utilised by many competitors.  killPlace is grouped by kills, and then sorted by winPlacePerc.  If we took all players with zero kills in one match and sorted by killPlace, their winPlacePerc would also be sorted in descending order.  

At this point, we should notice that every player of a group shares the same winPlacePerc, regardless of the position they were individually eliminated in.  This means we are only interested in ranking final position by groups, not players.
Let's imagine a hypothetical game, with a winning order of group A: 1st, B: 2nd, C: 3rd, D: 4th, E: 5th, F: 6th.  We have the following information.
Player 1: Group A, 1 kill, #4 killPlace
Player 2: Group A, 3 kills, #2 killPlace
Player 3: Group B, 4 kills, #1 killPlace
Player 4: Group C, 1 kill, #5 killPlace
Player 5: Group C, 0 kills, #7 killPlace
Player 6: Group D: 0 kills, #8 killPlace
Player 7: Group E, 2 kills, #3 killPlace
Player 8: Group E, 1 kill, #6 killPlace
Player 9: Group E, 0 kills, #9 killPlace
Player 10: Group F, 0 kills, #10 killPlace

Using this information, if we group by kills and sort by killPlace, we construct the below hierarchy:

	1.	0 kills:
	•	C
	•	D
	•	E
	•	F
	2.	1 kill:
	•	A
	•	C
	•	E
	3.	2 kills:
	•	E
	4.	3 kills:
	•	A
	5.	4 kills:
	•	B


The really powerful information is if we look at transitive relations - for example, if we look at 1 kill, we can see that A > C > E.  If we look at 0 kills, we see that C > D > E > F.  Because we know that A > C already (from the 1 kill data), then suddenly, we have learned virtually the final ranking of our game.  We know that A must also be better than D, E, and F.  Now, our order is A > C > D > E > F, and the only "unknown" group is B. 
This was the second key step of our model - finding the correct ordering for these "unknown" groups.  For this part, we used an XGB regression model, and predicted the winPlacePerc of each group (similar to what most people have been doing in kernels).  For this model, we incorporated some additional feature engineering.  We found the distance travelled by each group to be especially important, and engineered some features based off this.
In the above example, let's say that we predicted the winPlacePerc of group A to be 0.9, group B to be 0.85, and group C to be 0.6.  In this case, we would simply place group B between A and C.  Our final ranking then becomes the following: A > B > C > D > E > F.  Often in validation, we ranked many games perfectly, as the game was nearly fully sorted from killPlace alone.
To do this efficiently in the kernel, we constructed a directed acyclic graph for each game and then sorted it, using the XGB regression model to resolve any disputes.  This was surprisingly fast - it took only 7 minutes for all ~20K test games.
The third key step in our model was then post-processing.  In this competition, we were often missing groups from a game (when maxPlace > numGroups).  There has been an excellent kernel and discussion thread posted on this already.
Our solution was to assume that the missing ranks were uniformly distributed in a game.  Then, we would insert "fake" missing groups at even intervals within the game.  For example, if numGroups = 20 and maxPlace = 24 in a game, we would add a "fake" rank at ranks 4, 8, 12 and 16.  (In our above example, if numGroups = 6 and maxPlace = 7, we would insert the fake rank at rank 3.  Our solution would then look like A > B > C > ? > D > E > F).
If the missing ranks actually are randomly uniformly distributed, then this is the optimal solution for this problem.  However, there may have been further gains possible by attempting to predict what rank the missing groups finished.  Once the ranks were inserted, we converted the group rank back to winPlacePerc for each player.  This is a relatively simple step: (rank - max(rank)) / (min(rank) - max(rank)) for each game.
Thanks for a fun competition!  - Minions.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: