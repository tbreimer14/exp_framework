"""
Run a single individual from its genome in an output csv file.
Takes one command line arg "--gen" corresponding to generation number.
Takes another command line arg "--mode" which displays the simulation in different ways.
"--mode s" makes the simulation output to the screen, replacing it with "--mode v" saves 
each simulation as a video in `./videos`. "-mode b" shows on screen and saves a video.
Must also specify --filename for csv file.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import argparse
import pathlib
import pandas as pd
from snn_sim_two_corners import run as run_two_corners
from snn_sim_four_corners import run as run_four_corners

ITERS = 1000
GENOME_START_INDEX = 3
SIMULATION = "two-corners"

def run_indvididual(generation, mode, filename, sim):
    """
    Run an individual from a csv file.
    
    Parameters:
        generation (int): Generation number of individual.
        mode (string): Tells whether to show simulation, save it to
                       video, or both. "screen" renders the video to the screen. "video" saves a
                       video to the "./videos" folder. "both" does both of these things.
        filename (string): CSV file to look at. Should be in /data directory.
        sim (func): What function to use to run the simulation
    """

    if mode == "video" or mode == "both":
        os.makedirs("videos", exist_ok=True)

    this_dir = pathlib.Path(__file__).parent.resolve()
    df = pd.read_csv(os.path.join(this_dir, os.path.join("data", filename)))
    row = df.loc[(df['generation']==generation)]
    genome = row.values.tolist()[0][GENOME_START_INDEX:]

    vid_name = filename + "_gen" + str(generation)
    vid_path = os.path.join(this_dir, "videos")

    sim(ITERS, genome, mode, vid_name, vid_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument(
        '--mode', #headless, screen, video, both h, s, v, b
        help='mode for output. h-headless , s-screen, v-video, b-both',
        default="s")

    parser.add_argument(
        '--gen',
        type=int,
        help='what generation to grab',
        default=1)

    parser.add_argument(
        '--filename',
        type=str,
        help='what csv file to look at',
        default="default_csv.csv")
    
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

    run_indvididual(args.gen, args.mode, args.filename, fitness_fun)
