#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for reading configurations.

Author: Jan Schlüter
"""

import os
import io


def parse_variable_assignments(assignments):
    """
    Parses a list of key=value strings and returns a corresponding dictionary.
    Values are tried to be interpreted as float or int, otherwise left as str.
    """
    variables = {}
    for assignment in assignments or ():
        key, value = assignment.split('=', 1)
        for convert in (int, float, str):
            try:
                    value = convert(value)
            except ValueError:
                continue
            else:
                if isinstance(value, str) and value == 'False':
                    value = False
                elif isinstance(value, str) and value == 'True':
                    value = True
                break
        variables[key] = value
    return variables


def parse_config_file(filename):
    """
    Parses a file of key=value lines and returns a corresponding dictionary.
    Values are tried to be interpreted as float or int, otherwise left as str.
    Empty lines and lines starting with '#' are ignored.
    """
    with io.open(filename, 'r') as f:
        return parse_variable_assignments(
                [l.rstrip('\r\n') for l in f
                 if l.rstrip('\r\n') and not l.startswith('#')])


def write_config_file(filename, cfg):
    """
    Writes out a dictionary of configuration flags into a text file that is
    understood by parse_config_file(). Keys are sorted alphabetically.
    """
    # Again compatibility problems in python3 for writing a string as bytes into a file, therefore use w instead of wb
    with io.open(filename, 'w') as f:
        f.writelines("%s=%s\n" % (key, cfg[key]) for key in sorted(cfg))


def prepare_argument_parser(parser):
    """
    Adds suitable --vars and --var arguments to an ArgumentParser instance.
    """
    parser.add_argument('--vars', metavar='FILE',
            action='append', type=str,
            default=[os.path.join(os.path.dirname(__file__), 'models/defaults.vars')],
            help='Reads configuration variables from a FILE of KEY=VALUE '
                 'lines. Can be given multiple times, settings from later '
                 'files overriding earlier ones. Will read defaults.vars, '
                 'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE',
            action='append', type=str,
            help='Set the configuration variable KEY to VALUE. Overrides '
                 'settings from --vars options. Can be given multiple times.')


def from_parsed_arguments(options):
    """
    Read configuration files passed with --vars and immediate settings
    passed with --var from a given ArgumentParser namespace and returns a
    configuration dictionary.
    """
    cfg = {}
    for fn in options.vars:
        cfg.update(parse_config_file(fn))
    cfg.update(parse_variable_assignments(options.var))
    return cfg
