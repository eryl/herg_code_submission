from argparse import ArgumentParser
from pathlib import Path
from typing import List
from collections import defaultdict

import pandas as pd

from riseqsar.experiment.experiment_tracker import find_top_level_resamples

def predictor_performance_2_dfs(path_to_predictor: Path):
    performances = defaultdict(list)
    evaluation_dir = path_to_predictor / 'evaluation'
    
    if evaluation_dir.exists():
        for evaluation_folder in evaluation_dir.iterdir():
            performance_dir =  evaluation_folder / 'performance'

            model_name = path_to_predictor.parent.parent.name
            timestamp =  path_to_predictor.parent.parent.parent.name
            resample = path_to_predictor.name
            for performance_file in performance_dir.iterdir():
                performance = pd.read_csv(performance_file)
                performance['Model'] = pd.Series([model_name])
                performance['Timestamp'] = pd.Series([timestamp])
                performance['Resample'] = pd.Series([resample])
                performance['Performance file'] = pd.Series([performance_file.name])
                performances[evaluation_folder.name].append(performance)
    return performances

def summarize_predictors_from_runs(paths: List[Path], output_dir: Path = None):
    if output_dir is None:
        output_dir = Path()

    tag_performances = defaultdict(list)    
    
    for path_to_run in paths:
        top_level_resamples = find_top_level_resamples(path_to_run)

        for resample_dir in top_level_resamples:
            performances = predictor_performance_2_dfs(resample_dir)
            for tag, performance in performances.items():
                tag_performances[tag].extend(performance)

    for tag, performances in tag_performances.items():
        if performances:
            performance_dataframe = pd.concat(performances)
            output_file = output_dir / f'performance_{tag}.csv'
            performance_dataframe.to_csv(output_file, index=False)
        else:
            print(f"No performance for tag {tag}")
        


def main():
    parser = ArgumentParser(description="Script for summarizing model performance")
    parser.add_argument('experiment_roots', help="Root directories to search for experimetns in", nargs='+', type=Path)
    parser.add_argument('--output-dir', help="Where to save the performance summaries", type=Path)
    args = parser.parse_args()

    summarize_predictors_from_runs(args.experiment_roots, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
    
    
