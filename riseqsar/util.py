import json
from dataclasses import is_dataclass
import copy
import datetime
import importlib.util
import re
from collections import Counter
from pathlib import Path
from typing import TypeVar, Union, Type, Optional, Sequence


def timestamp():
    """
    Generates a timestamp.
    :return:
    """
    t = datetime.datetime.now().replace(microsecond=0)
    #Since the timestamp is usually used in filenames, isoformat will be invalid in windows.
    #return t.isoformat()
    # We'll use another symbol instead of the colon in the ISO format
    # YYYY-MM-DDTHH:MM:SS -> YYYY-MM-DDTHH.MM.SS
    time_format = "%Y-%m-%dT%H.%M.%S"
    return t.strftime(time_format)


def sliding_window(x, window_length, step_length=1, axis=0):
    """Use stride tricks on the array x to generate a sliding window over the submitted axis"""
    from numpy.lib.stride_tricks import as_strided
    if step_length > 1:
        raise NotImplementedError("Not implemented for step sizes greater than 1")
    axis_shape = x.shape[axis]
    axis_stride = x.strides[axis]
    n_windows = (
                        axis_shape - window_length + 1) // step_length  # To support step sizes greater than 1 we also need to
    # slice the array so it's an even size along the chosen
    # axis. Perhaps we should force the user to supply valid arrays, step length and window sizes
    # The stride along each element in a window should be the same as the current stride of the axis
    # the stride between windows should be the same times the step size
    in_window_stride = axis_stride
    between_window_stride = axis_stride * step_length
    new_shape = x.shape[:axis] + (n_windows, window_length) + x.shape[axis + 1:]
    new_stride = x.strides[:axis] + (between_window_stride, in_window_stride) + x.strides[axis + 1:]
    x2 = as_strided(x, shape=new_shape, strides=new_stride)
    return x2


def load_module(module_path: Union[str, Path]):
    """
    Loads a python module with the given module path
    :param module_path: Path of module file to load
    :return: Module object
    """
    spec = importlib.util.spec_from_file_location("module_from_file", module_path)
    module_from_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_from_file)
    return module_from_file

T = TypeVar('T')
def load_config(training_config_path: Path, config_type: Type[T], default: Optional[Type[T]]=None) -> T:
    """
    Given a file path, load it as a module and return the first matching object of *config_type*
    :param training_config_path: File containing the module to load
    :param config_type: The object type to look for. E.g. a custom object or dataclass instance
    :param default: If an instance of the desired class could not be found, return this value instead.
    :return: The first found instance of *config_type*. If no instance is found, ValueError is raised
    """
    mod = load_module(training_config_path)
    for k,v in mod.__dict__.items():
        if isinstance(v, config_type):
            return v
    if default is not None:
        return default
    else:
        raise ValueError(f"File {training_config_path} does not contain any attributes of type {config_type}")


def reconstruct_dataclass(str_rep, dc):
    pattern = r"{}\((.*)\)".format(dc.__name__)
    m = re.match(pattern, str_rep)
    if m is not None:
        args, = m.groups()
        # We need to sanitize the string, converting any <Object at 0x0000> to a string
        sanitizers = [r"<([\w.]+) object at 0x[\da-f]+>", r"<class '([\w.]+)'>"]
        replaced = r"'\g<1>'"
        sanitized = args
        for sanitizer in sanitizers:
            sanitized = re.sub(sanitizer, replaced, sanitized)
        contents = eval('dict({})'.format(sanitized))
        return contents


class JSONEncoder(json.JSONEncoder):
    """Custom JSONEncoder which tries to encode failed types (like pathlib Paths) as strings, also adds special
    handling of dataclasses, adding a 'dataclass_name' and 'dataclass_module' field"""

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

def common_prefix(*paths: Sequence[Path]):
    """Common prefix of given paths"""
    counter = Counter()

    for path in paths:
        assert isinstance(path, Path)
        counter.update([path])
        counter.update(path.parents)

    try:
        return sorted((x for x, count in counter.items() if count >= len(paths)), key=lambda x: len(str(x)))[-1]
    except LookupError as e:
        raise ValueError('No common prefix found') from e


def listify(collection):
    '''Convert a possible nested collection to a 1D list'''
    try:
        return [x for sublist in collection for x in listify(sublist)]
    except TypeError:
        return [collection]