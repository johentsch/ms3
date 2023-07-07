import hashlib
import json
import os
import re
from ast import literal_eval
from base64 import urlsafe_b64encode
from functools import cache
from pprint import pprint
from typing import Optional, Tuple, Iterable

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
        elif string_converter == literal_eval:
            field['type'] = 'array'
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
            raise NotImplementedError(f"Don't know how to handle pandas dtype {pandas_dtype}")
    else:
        raise NotImplementedError(f"Don't know how to handle column {column_name}")
    if len(constraints) > 0:
        field['constraints'] = constraints
    return field


def make_frictionless_schema_descriptor(column_names: Tuple[str, ...],
                                        primary_key: Optional[Tuple[str, ...]] = None,
                                        **custom_data
                                        ) -> dict:
    fields = []
    for column_name in column_names:
        field = column_name2frictionless_field(column_name)
        if not "type" in field:
            raise ValueError(f"column_name2frictionless_field({column_name!r}) = {field} (missing 'type'!)")
        fields.append(field)
    descriptor = dict(fields=fields)
    if primary_key:
        descriptor['primaryKey'] = primary_key
    if len(custom_data) > 0:
        descriptor.update(custom_data)
    return descriptor

FRICTIONLESS_REGEX = r"^([-a-z0-9._/])+$"
FRICTIONLESS_INVERSE = r"[^-a-z0-9._/]"
schemata = {}

def make_resource_name(name: str, replace_char="_") -> str:
    name = name.lower()
    if not re.match(FRICTIONLESS_REGEX, name):
        name = re.sub(FRICTIONLESS_INVERSE, replace_char, name)
    return name

def assemble_resource_descriptor(name: str,
                                 path: str,
                                 schema: str,
                                 innerpath: Optional[str] = None,
                                 ):
    is_zipped = path.endswith(".zip")
    if is_zipped:
        assert innerpath is not None, "Must specify innerpath for zip files."
    descriptor = {
        "name": make_resource_name(name),
        "type": "table",
        "path": path,
        "scheme": "file",
        "format": "tsv",
        "mediatype": "text/tsv",
    }
    if is_zipped:
        descriptor["compression"] = "zip"
        descriptor["innerpath"] = innerpath
    descriptor.update({
        "encoding": "utf-8",
        "dialect": {
            "csv": {
                "delimiter": "\t"
            }
        },
        "schema": schema
    })
    return descriptor


def replace_extension(filepath: str, new_extension: str) -> str:
    if new_extension[0] != '.':
        new_extension = '.' + new_extension
    return os.path.splitext(filepath)[0] + new_extension

def make_json_path(file: File) -> str:
    return os.path.join(file.directory, f"{file.fname}.resource.json")

UTILS_DIR = os.path.dirname(__file__) # .../ms3/src/ms3/utils/
SCHEMAS_DIR = os.path.join(UTILS_DIR, '..', '..', '..', 'schemas')
os.makedirs(SCHEMAS_DIR, exist_ok=True)

# SCHEMA_REGISTRY_PATH = os.path.join(SCHEMAS_DIR, 'schema_registry.json')
# SCHEMA_REGISTRY: dict = None
#
#
# def get_schema_registry() -> dict:
#     global SCHEMA_REGISTRY
#     if SCHEMA_REGISTRY is None:
#         SCHEMA_REGISTRY = dict()
#         if os.path.exists(SCHEMA_REGISTRY_PATH):
#             with open(SCHEMA_REGISTRY_PATH, 'r') as f:
#                 SCHEMA_REGISTRY = json.load(f)
#     return SCHEMA_REGISTRY

# def register_new_schema_locally(descriptor: dict):
#     registry = get_schema_registry()
#     facet = descriptor['facet']
#     identifier = descriptor['identifier']
#     filename = descriptor['filename']
#     if facet in registry:
#         if identifier in registry[facet]:
#             raise ValueError(f"Schema {facet}/{identifier} already registered!")
#         registry[facet][identifier] = filename
#     else:
#         registry[facet] = {identifier: filename}
#     with open(SCHEMA_REGISTRY_PATH, 'w') as f:
#         json.dump(registry, f, indent=2)

def get_truncated_hash(S: str | Iterable[str],
                       hash_func = hashlib.sha1,
                       length = 10) -> str:
    """Computes the given hashfunction for the given string(s), and truncates the result."""
    if isinstance(S, str):
        S = [S]
    hasher = hash_func()
    for s in S:
        hasher.update(s.encode('utf-8'))
    return urlsafe_b64encode(hasher.digest()[:length]).decode('utf-8').rstrip('=')






def get_schema_or_url(facet: str,
                      column_names: Tuple[str],
                      index_levels: Optional[Tuple[str]] = None,
                      base_local_path = SCHEMAS_DIR,
                      base_url = "https://raw.githubusercontent.com/johentsch/ms3/main/schemas/",
                      ) -> str | dict:
    """ Given a facet name (=subfolder) and a tuple of [index column names +] column names, compute an identifier and
    if that schema exists under ``<base_url>/<facet>/<identifier>.schema.yaml`` return that URL, or otherwise
    create a the frictionless schema descriptor based on the column names, using the descriptions and types the ms3
    stores for known column names (see :func:`make_frictionless_schema_descriptor`) and treating unknown columns
    as string fields. In the latter case, the YAML file is written to ``<base_local_path>/<facet>/<identifier>.schema.yaml``
    so specify local and remote bases such that the latter can be easily updated with the former.
    Both types of function outputs can be used for the ``schema`` key in a frictionless table resource descriptor.

    Args:
        facet: Name of the subfolder where to store the schema descriptor.
        column_names:
            Names of the schema fields. Used for generating the hash-based identifier and for creating the actual
            frictionless descriptor based on known field names. Unknown field names are assumed to be strings.
        index_levels:
            Additional index column names prepended to the column names. They are specified separately because in the
            frictionless schema they are declared under ``primaryKey``, meaning that the IDs will be validated on the
            basis of being required and unique.
        base_local_path:
            Schema descriptors will be created locally under ``<base_local_path>/<facet>/<identifier>.schema.yaml``,
            unless the schema is found online.
        base_url:
            Schema descriptor is found at``<base_url>/<facet>/<identifier>.schema.yaml``, the function returns the
            URL rather than the descriptor dict.

    Returns:

    """
    if base_url[-1] != '/':
        base_url += '/'
    if index_levels is None:
        column_names = tuple(column_names)
    else:
        column_names = tuple(index_levels) + tuple(column_names)
    schema_identifier = get_truncated_hash(column_names)
    schema_filename = f"{schema_identifier}.schema.yaml"
    schema_filepath = f"{facet}/{schema_filename}" # for URL & uniform filepath
    schema_path = os.path.join(base_local_path, facet, schema_filename) # for local OS
    if os.path.exists(schema_path):
        schema_url = f"{base_url}{schema_filepath}"
        # check if URL exists
        try:
            fl.Schema(schema_url)
            return schema_url
        except fl.FrictionlessException:
            try:
                schema = fl.Schema(schema_path)
                return schema.to_descriptor()
            except fl.FrictionlessException:
                descriptor = json.load(open(schema_path, 'r'))
                pprint(descriptor)
                raise
    else:
        descriptor = make_frictionless_schema_descriptor(column_names=column_names,
                                                         primary_key=index_levels,
                                                         # the rest is custom data added to the schema descriptor
                                                         facet=facet,
                                                         identifier=schema_identifier,
                                                         filepath=schema_filepath,
                                                         )
        fl.Schema(descriptor).to_yaml(schema_path)
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
def store_descriptor_and_validate(df, file_path, facet, fname) -> fl.Report:
    schema = get_schema(df, facet)
    json_path = replace_extension(file_path, '.resource.json')
    store_descriptor(fname, facet, schema, json_path, logger=logger)
    return fl.validate(json_path)