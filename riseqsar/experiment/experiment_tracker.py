import csv
import itertools
import shutil
import copy
from collections import Mapping, defaultdict
from dataclasses import dataclass, is_dataclass
from typing import Optional
import json
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import dill


CHILD_DIRECTORY = 'children'

class JSONEncoder(json.JSONEncoder):
    "Custom JSONEncoder which tries to encode filed types (like pathlib Paths) as strings"
    def default(self, o):
        if is_dataclass(o):
            attributes = copy.copy(o.__dict__)
            attributes['dataclass_name'] = o.__class__.__name__
            attributes['dataclass_module'] = o.__module__
            return attributes
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError:
            return str(o)

@dataclass
class Event:
    content: str
    id: int
    timestamp: datetime.datetime


class ExperimentTracker(object):
    def __init__(self, output_dir: Path, identifier=None, parent=None, tag=None, starting_count=0):
        self.output_dir = output_dir
        self.parent = parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = open(self.output_dir / 'events.txt', 'w')
        self.event_id = 0
        self.events = dict()
        self.metadata_path = self.output_dir / 'experiment_metadata'
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.children = []
        self.count = starting_count
        self.identifier = identifier
        self.tag = tag

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.event_log.close()

    def log_event(self, event: str, reference_event: Optional[Event] = None) -> Event:
        t = datetime.datetime.now().replace(microsecond=0)
        event_id = self.event_id
        self.event_id += 1
        event = Event(content=event, id=event_id, timestamp=t)
        self.events[event_id] = event
        
        timestamp = t.isoformat()
        if reference_event is not None:
            timedelta = t - reference_event.timestamp
            # We remove the microseconds for clarity
            timedelta = str(datetime.timedelta(seconds=round(timedelta.total_seconds())))
            self.event_log.write(f'{event_id:02} {timestamp} [{timedelta} since event {reference_event.id}]:\t{event.content}\n')
        else:
            self.event_log.write(f'{event_id:02} {timestamp}:\t{event.content}\n')
        self.event_log.flush()
        return event

    def log_artifact(self, name, artifact):
        artifacts_path = self.output_dir / 'artifacts'
        artifacts_path.mkdir(exist_ok=True)
        with open(artifacts_path / f'{name}.pkl', 'wb') as fp:
            dill.dump(artifact, fp)

    def log_artifacts(self, artifacts):
        if isinstance(artifacts, Mapping):
            artifacts = artifacts.items()
        for artifact_name, artifact in artifacts:
            self.log_artifact(artifact_name, artifact)

    def log_file(self, src_path: Path, dst_path=None):
        files_path = self.output_dir / 'files'
        files_path.mkdir(exist_ok=True)
        if dst_path is None:
            dst_path = src_path.name
        dst_path = files_path / dst_path
        shutil.copy(src_path, dst_path)

    def log_files(self, files):
        if isinstance(files, Mapping):
            files = [(src, dst) for dst, src in files.items()]
        for item in files:
            try:
                src, dst = item
                self.log_file(src, dst)
            except ValueError:
                self.log_file(item)

    def log_scalar(self, name, value):
        scalars_dir = self.output_dir / 'scalars'
        scalars_dir.mkdir(exist_ok=True)
        scalars_path = scalars_dir / (name + '.csv')
        if not scalars_path.exists():
            with open(scalars_path, 'a') as fp:
                fp.write('timestamp,value\n')
                fp.write(f'{self.count},{value}\n')
        else:
            with open(scalars_path, 'a') as fp:
                fp.write(f'{self.count},{value}\n')

    def log_scalars(self, scalars):
        if isinstance(scalars, Mapping):
            scalars = scalars.items()
        for name, value in scalars:
            self.log_scalar(name, value)

    def log_vector(self, name, value):
        raise NotImplementedError('log_vector has not been implemented')

    def log_model(self, model_reference, model_object):
        model_directory = self.output_dir / 'models'
        model_directory.mkdir(exist_ok=True)
        model_factory, model_state = model_object.serialize(model_directory, tag=model_reference)
        model_path = model_directory / (model_reference + '.pkl')
        with open(model_path, 'wb') as fp:
            dill.dump(dict(model_factory=model_factory,
                           model_state=model_state), fp)
        self.make_reference(model_reference, model_path)

    def log_json(self, name, value):
        jsons_dir = self.output_dir / 'jsons'
        jsons_dir.mkdir(exist_ok=True)
        json_file_path = jsons_dir / (name + '.json')
        with open(json_file_path, 'w') as fp:
            json.dump(value, fp, cls=JSONEncoder)

    def tablify_values(self, values):
        if isinstance(values, Mapping):
            try:
                values = pd.DataFrame(values)
            except ValueError:
                # For now we assume the problem is that each "column" is scalar
                values = pd.DataFrame({k: [v] for k, v in values.items()})
        return values

    def log_table(self, name, values):
        tables_path = self.output_dir / 'tables'
        tables_path.mkdir(exist_ok=True)
        csv_path = tables_path / (name + '.csv')
        values = self.tablify_values(values)
        if isinstance(values, pd.DataFrame):
            values.to_csv(csv_path, index=False)

    def log_performance(self, dataset_name, values, tag=None):
        tables_path = self.output_dir / 'evaluation' / dataset_name / 'performance'
        tables_path.mkdir(exist_ok=True, parents=True)

        if tag is not None:
            csv_path = tables_path / f'{tag}_performance.csv'
        else:
            csv_path = tables_path / f'performance.csv'

        values = self.tablify_values(values)
        if isinstance(values, pd.DataFrame):
            values.to_csv(csv_path, index=False)

    def log_predictions(self, dataset_name, values, tag=None):
        tables_path = self.output_dir / 'evaluation' / dataset_name / 'predictions'
        tables_path.mkdir(exist_ok=True, parents=True)

        if tag is not None:
            csv_path = tables_path / f'{tag}_predictions.csv'
        else:
            csv_path = tables_path / f'predictions.csv'

        values = self.tablify_values(values)
        if isinstance(values, pd.DataFrame):
            values.to_csv(csv_path, index=False)


    def log_numpy(self, name, value):
        numpy_dir = self.output_dir / 'ndarrays'
        numpy_dir.mkdir(exist_ok=True)

        if isinstance(value, dict):
            np.savez(numpy_dir / name, value)
        else:
            np.save(numpy_dir / name, value)

    def reference_exists(self, reference):
        "Check if a reference exists"
        references_dir = self.output_dir / 'references'
        references_path = references_dir / (reference + '.pkl')
        return references_path.exists()

    def lookup_reference(self, reference):
        references_dir = self.output_dir / 'references'
        references_path = references_dir / (reference + '.pkl')
        with open(references_path, 'rb') as fp:
            value = dill.load(fp)
            return value

    def delete_model(self, model_reference):
        model_path = self.lookup_reference(model_reference)
        model_path.unlink()
        self.remove_reference(model_reference)

    def make_reference(self, reference_name, referred_value):
        references_dir = self.output_dir / 'references'
        references_dir.mkdir(exist_ok=True)
        references_path = references_dir / (reference_name + '.pkl')
        with open(references_path, 'wb') as fp:
            dill.dump(referred_value, fp)

    def remove_reference(self, reference_name):
        references_dir = self.output_dir / 'references'
        references_dir.mkdir(exist_ok=True)
        references_path = references_dir / (reference_name + '.pkl')
        references_path.unlink()

    def make_child(self, child_directory=None, tag=None):
        if tag is None:
            i = len(self.children)
            tag = f'child_{i}'
        if child_directory is None:
            child_directory = CHILD_DIRECTORY
        # NOTE: if you change this, make sure you change the find_experiments function below
        children_dir = self.output_dir / child_directory
        children_dir.mkdir(exist_ok=True)
        child_dir = children_dir / tag
        child_tracker = ExperimentTracker(child_dir, parent=self, tag=tag)
        self.children.append((child_dir, tag))
        return child_tracker

    def get_json(self, name):
        jsons_dir = self.output_dir / 'jsons'
        json_file_path = jsons_dir / (name + '.json')
        with open(json_file_path, 'r') as fp:
            value = json.load(fp)
        return value

    def load_model(self, reference, force_cpu=False):
        model_path = self.lookup_reference(reference)
        with open(model_path, 'rb') as fp:
            model_dict = dill.load(fp)
            model_factory = model_dict['model_factory']
            model_state = model_dict['model_state']
            model = model_factory(model_state)
            if force_cpu:
                model.set_device('cpu')
            return model

    def progress(self, n=1):
        self.count += n

    def get_identifier(self):
        if self.identifier is not None:
            return self.identifier
        else:
            return self.output_dir.name

    def get_progenitor(self):
        """Return the root level ExperimentTracker"""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

def find_experiment_top_level_models(path: Path):
    "Finds all top level experiments with logged models in the given path"
    # Perhaps not the best way of ensuring we get the root experiments is to check common prefixes for all folders with
    # a experiment_metadata subdirectory
    experiments_paths = set([p.parent for p in path.glob('**/models')])
    path_prefixes = defaultdict(list)
    for p in experiments_paths:
        for prefix in itertools.accumulate(p.parts, lambda a, b: Path(a) / b):
            if prefix in experiments_paths:
                path_prefixes[prefix].append(p)
                break

    fixed_experiment_paths = sorted(path_prefixes.keys())
    return fixed_experiment_paths


def find_top_level_resamples(path: Path):
    "Finds all top level resamples in the given path"
    
    experiments_paths = set(path.glob('**/resample_*'))
    path_prefixes = defaultdict(list)
    
    # This code looks for the shortest prefix, gradually constructing the whole path, 
    # but aborting when it finds a match in experiment_paths. 
    # This keeps nested resamples from showing up
    for p in experiments_paths:
        # accumulate will gradually build up the path until 
        # the result matches a path in experiment paths. This means that paths 
        # which are children to some path in experiment_path will be filtered out.
        for prefix in itertools.accumulate(p.parts, lambda a, b: Path(a) / b):
            if prefix in experiments_paths:
                # If there is a match for the currenly built prefix in 
                # experiment_paths, we break here. This stops nested 
                # resamples from showing up
                path_prefixes[prefix].append(p)
                break

    fixed_experiment_paths = sorted(path_prefixes.keys())
    return fixed_experiment_paths