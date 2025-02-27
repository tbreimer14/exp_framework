# exp-framework

Implementation of an experimental framework for evolving SNN-controlled robots in evogym.

## Example commands

See example_commands.md for examples on how to run everything.

## `run_cmaes.py`

Runs cma-es with an closed loop SNN-controlled robot. Creates and output csv file in /data and updates it continuously with the best individual from each generation. Whether to show the simulation or save as video, number of generations, sigma can be passed as command line arguments. Example: `python3 run_cmaes.py --mode h --gens 50 --sigma 2` runs cma-es for 50 generations in headless mode with a sigma of 2. Replacing "--mode h" with "--mode s" shows the best individual from each generation, and replacing it with "--mode v" saves the best individual from each generation as a video in `./videos` "--mode b" shows on screen and saves a video.

## `best_individual.py`

Once `run_cmaes.py` is running, `python3 best_individual.py --file CSV_FILE_NAME.csv` can be run simultaneously, which plays the best found individual so far in a loop from the specified csv file.

## `run_individual.py` 

Runs a single individual from its genome in an output csv file. Takes `--generation GEN_NUM` `--mode MODE_CHAR` command line args.

## `plot_genome.py`

Plots genome values on the x and y axis and color based on fitness. Takes three command line arguments: --filename, --x-axis, --y-axis. This is intended to show the relationship between two genome values and how it affects fitness.

## `plot_fitness_over_gens.py`

Given a csv file as a command line argument, plot fitness over generations.
