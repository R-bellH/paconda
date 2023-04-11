[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Pacman project
<p align="center">
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/el/0/00/Pac-Man.png"> <br />
</p><br />

## Description
this project focuses on the implementation of AI algorithms that treat the world as continuous space and navigate it in order to find the shortest path to a goal, in our case the ghosts try to find the shortest path to Pacman in order to capture it. 

The project is based on the Pacman game and the simulator is provided by UC Berkeley.

we've implemented the following algorithms:
1. PRM (Probabilistic Roadmap) with Dijkstra algorithm and A* algorithm
2. RRT (Rapidly-exploring Random Tree) and RRT* with Dijkstra algorithm
3. Grid based algorithms: BFS, A* ***(?)***

all algorithms have been implemented from scratch and have the required infrastructure to work with heuristics.

## Navigating the project
most of the files in the project were untouched and are provided by UC Berkeley for the simulation, the files that were modified are:
1. 'ghostAgents.py' - contains the implementation of the algorithms discussed in the pdf
   - each algorithm is implemented as a class in the ghostAgents.py file:
      - PRMGhost
      - RRTGhost
      - AStarGhost
      - GridGhost
      - FlankGhost
2. 'utils.py' - added some functionalities (such as Bresenham) that we needed for the algorithms
3. 'pacman.py' - added a way to use different agents for the ghosts in the same game
additionaly we've added several completely new files:
1. 'show.py','show_*' - files that are used to visualize the algorithms
   - show - visualize several ghosts at one run
   - show_PRM - visualize the PRM ghost by showing the graph and the path
   - show_RRT - visualize the RRT ghost by showing the graph and the path
   - show_Grid - visualize the Grid ghost by building a gif showing the building of the map the ghost know of
2. 'PRM.py' - a graph class that we built to implement the graph for both the PRM and the RRT, the graph have some added functionality

for more information and hyperparameters for the ghosts look at each ghost implementation in ghostAgent.py 

lastly we've added a folder Media that contains all the gifs visualization that we've created. as well as some additional videos.

### Running the project
##### TL;DR
1. run pacman.py
2. run show_PRM.py

##### full walkthrough and explanation
if you want to run the game with the default agents and map you can simply run pacman.py with no arguments
you can also customize these setting by adding arguments to the command line:
1. '-p' - the pacman agent, the default is KeyboardAgent
2. '-g' - the ghosts agent, the default is RRTGhost
   - if you want to try a different agent you can use the following:
     1. '-g PRMGhost' - the PRM algorithm
     2. '-g RRTGhost' - the RRT algorithm
     3. '-g AStarGhost' - the A* algorithm
     4. '-g GridGhost' - the Grid algorithm
     5. '-g FlankGhost' - the Flank algorithm
     6. '-g RandomGhost' - a ghost that move randomly (sort of baseline)
   - do note that running more than one kind of ghost agent in the same game requires further fiddling with the code (more on line 558 in pacman.py)
3. '-l' - the map, the default is smallOpening
   - full list of maps can be found in the 'layouts' folder
   - not all maps support more than one ghost (and some not even that)
4. '-k' - the maximum number of ghosts to use, the default is 1
- example: 'python pacman.py -p KeyboardAgent -g PRMGhost -l originalClassic -k 2'

#### visualize the algorithms
you can visualize each ghost by running the appropriate show file after you had a run of the game. for example to visualize the PRM algorithm you can run show_PRM.py 
this will result in the final planned path for the ghost to capture pacman. and its final map.
alternatively you can run show_PRM.py *at the same time* you run the game which will show you the ghost current map and plan for the time you run the command.
- note that show_Grid.py work a little differently as it produces gif instead of graph as we felt it better represent the algorithm work.

showing the graphs as described here will only show you the graph of the first ghost. for more ghosts you will need edit the show.py to show you the ghosts that you have chosen.

```
## Links for more information regarding the simulation and Berkeley's original project
1. http://ai.berkeley.edu/search.html
2. http://ai.berkeley.edu/multiagent.html
