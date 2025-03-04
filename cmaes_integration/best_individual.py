"""
Visualize the best individual so far, pulling from output.csv.

Author: James Gaskell
February 6th, 2025
"""

import os
import argparse
import pandas
from snn_sim_two_corners import run as run_two_corners
from snn_sim_four_corners import run as run_four_corners

ITERS = 1000
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATION = "two-corners"

def visualize_best(filename, sim):
    """
    Look at output.csv and continuously run best individual.
    Assumes csv names are their best achieved fitnesses
    Continually searches for the lowest best fitness, plays the visualization and repeats
    """

    while True:
        path = os.path.join(ROOT_DIR, "data", filename)
        df = pandas.read_csv(path)
        best_fitness = min(df["best_fitness"])
        row = df.loc[df['best_fitness'] == best_fitness]
        genome = row.values.tolist()[0][3:]
        sim(ITERS, genome, "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--file',
                        type=str,
                        default=None,
                        help='csv file to run')
    
    parser.add_argument('--sim',
                        type=str,
                        default=SIMULATION,
                        help='specify what simulaion to use: two-corners, four-corners')
    args = parser.parse_args()

    if args.sim == "two-corners":
        fitness_fun = run_two_corners.run
    elif args.sim == "four-corners":
        fitness_fun = run_four_corners.run
    else:
        raise RuntimeError("Unknown simulation! Try 'two-corners' or 'four-corners'")
    
    args = parser.parse_args()

    if args.file == None:
        raise Exception('No csv file specified!')

    visualize_best(args.file)
