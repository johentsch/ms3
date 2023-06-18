import json
import os
import re
from functools import cache
from pprint import pformat
from typing import Sequence

import frictionless as fl
import pandas as pd

from ms3._typing import ScoreFacet
from ms3.utils import TSV_COLUMN_TITLES, TSV_COLUMN_DESCRIPTIONS, TSV_DTYPES, TSV_COLUMN_CONVERTERS, function_logger, safe_frac, safe_int, str2inttuple, int2bool, File

FIELDS_WITHOUT_MISSING_VALUES = (
    'mc',
    'mc_playthrough',
)
FRACTION_REGEX = r"\d+(?:\/\d+)?" # r"-?\d+(?:\/\d+)?" for including negative fractions
INT_ARRAY_REGEX = r"^[([]?(?:-?\d+\s*,?\s*)*[])]?$"


@cache
def column_name2frictionless_field(column_name) -> dict:
    global FRACTION_REGEX, INT_ARRAY_REGEX
    field = dict(
        name = column_name,
    )
    if column_name in FIELDS_WITHOUT_MISSING_VALUES:
        constraints = dict(required=True)
    else:
        constraints = dict()
    title = TSV_COLUMN_TITLES.get(column_name)
    description = TSV_COLUMN_DESCRIPTIONS.get(column_name)
    pandas_dtype = TSV_DTYPES.get(column_name, str)
    string_converter = TSV_COLUMN_CONVERTERS.get(column_name)
    if title:
        field['title'] = title
    if description:
        field['description'] = description
    if string_converter is not None:
        pass
    if string_converter:
        if string_converter == safe_frac:
            field['type'] = 'string'
            constraints["pattern"] = FRACTION_REGEX
        elif string_converter == safe_int:
            field['type'] = 'integer'
            field['bareNumber'] = False # allow other leading and trailing characters
        elif string_converter == str2inttuple:
            field['type'] = 'string'
            constraints["pattern"] = INT_ARRAY_REGEX
        elif string_converter == int2bool:
            field['type'] = 'boolean'
        else:
            NotImplementedError(f"Unfamiliar with string converter {string_converter}")
    elif pandas_dtype:
        if pandas_dtype in (int, 'Int64'):
            field['type'] = 'integer'
        elif pandas_dtype == float:
            field['type'] = 'number'
        elif pandas_dtype in (str, 'string'):
            field['type'] = 'string'
        else:
            NotImplementedError(f"Don't know how to handle pandas dtype {pandas_dtype}")
    else:
        NotImplementedError(f"Don't know how to handle column {column_name}")
    if len(constraints) > 0:
        field['constraints'] = constraints
    return field


def make_frictionless_descriptor(column_names: Sequence[str]) -> dict:
    fields = []
    for column_name in column_names:
        field = column_name2frictionless_field(column_name)
        fields.append(field)
    descriptor = dict(fields=fields)
    return descriptor

FRICTIONLESS_REGEX = r"^([-a-z0-9._/])+$"
FRICTIONLESS_INVERSE = r"[^-a-z0-9._/]"
schemata = {}

def make_resource_name(name: str, replace_char="_") -> str:
    name = name.lower()
    if not re.match(FRICTIONLESS_REGEX, name):
        name = re.sub(FRICTIONLESS_INVERSE, replace_char, name)
    return name

def assemble_resource_descriptor(name, path, schema):
    return {
        "name": make_resource_name(name),
        "type": "table",
        "path": path,
        "scheme": "file",
        "format": "tsv",
        "mediatype": "text/tsv",
        "encoding": "utf-8",
        "dialect": {
            "csv": {
                "delimiter": "\t"
            }
        },
        "schema": schema
    }

def replace_extension(filepath: str, new_extension: str) -> str:
    if new_extension[0] != '.':
        new_extension = '.' + new_extension
    return os.path.splitext(filepath)[0] + new_extension

def make_json_path(file: File) -> str:
    return os.path.join(file.directory, f"{file.fname}.resource.json")

SCHEMAS_DIR = os.path.join(os.path.dirname(__file__), 'schemas')

def get_schema_or_url(facet: str, column_names: Sequence[str],):
    column_initials = ''.join(str(col)[0] for col in column_names)
    schema_name = f"{facet}_{column_initials}"
    schema_filename = f"{schema_name}.schema.json"
    schema_path = os.path.join(SCHEMAS_DIR, schema_filename)
    if os.path.exists(schema_path):
        schema_url = f"https://raw.githubusercontent.com/johentsch/ms3/main/src/ms3/utils/schemas/{schema_filename}"
        # check if URL exists
        try:
            fl.Schema(schema_url)
            return schema_url
        except fl.FrictionlessException:
            schema = fl.Schema(schema_path)
            return schema.to_descriptor()
    else:
        descriptor = make_frictionless_descriptor(column_names=column_names)
        fl.Schema(descriptor).to_json(schema_path)
        return descriptor


def get_schema(df: pd.DataFrame,
               facet: ScoreFacet,
               fl_infer: bool = False) -> dict | str:
    if fl_infer:
        result = fl.describe(df).schema.to_descriptor()
    else:
        result = get_schema_or_url(facet=facet, column_names=df.columns)
    return result

@function_logger
def store_descriptor(fname: str,
                     facet: ScoreFacet,
                     schema,
                     json_path: str
                     ) -> dict:
    descriptor = assemble_resource_descriptor(name=f"{fname}.{facet}",
                                              path=f"{fname}.tsv",
                                              schema=schema
                                              )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(descriptor, f, indent=2)
    logger.debug(f"Stored {json_path}")
    return descriptor

@function_logger
def store_descriptor_and_validate(df, file_path, facet, fname):
    schema = get_schema(df, facet)
    json_path = replace_extension(file_path, '.resource.json')
    store_descriptor(fname, facet, schema, json_path, logger=logger)
    validate_descriptor(json_path, logger=logger)

@function_logger
def validate_descriptor(path):
    report = fl.validate(path)
    if not report.valid:
        error_report = pformat(report)
        raise fl.FrictionlessException(error_report)
    logger.info(f"Frictionless descriptor successfully validated: {path}")