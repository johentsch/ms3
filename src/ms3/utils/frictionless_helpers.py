import hashlib
import json
import os
import re
from ast import literal_eval
from base64 import urlsafe_b64encode
from functools import cache
from pprint import pprint
from typing import Optional, Tuple, Iterable, Literal

import frictionless as fl
import pandas as pd
import yaml

from ms3._typing import ScoreFacet, TSVtypes, TSVtype
from .functions import TSV_COLUMN_TITLES, TSV_COLUMN_DESCRIPTIONS, TSV_DTYPES, TSV_COLUMN_CONVERTERS, function_logger, safe_frac, safe_int, str2inttuple, int2bool, File, \
    eval_string_to_nested_list, write_tsv, resolve_facets_param
from ms3.logger import function_logger

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
        elif string_converter in (literal_eval, eval_string_to_nested_list):
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
    if primary_key:
        for ix_level, column in zip(primary_key, column_names):
            if ix_level != column:
                raise ValueError(f"primary_key {primary_key} does not match column_names {column_names[:len(primary_key)]}")
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

def make_valid_frictionless_name(name: str, replace_char="_") -> str:
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, not {type(name)}")
    name = name.lower()
    if not re.match(FRICTIONLESS_REGEX, name):
        name = re.sub(FRICTIONLESS_INVERSE, replace_char, name)
    return name

@function_logger
def assemble_resource_descriptor(
        resource_name: str,
        filepath: str,
        schema: str | dict,
        innerpath: Optional[str] = None,
):
    is_zipped = filepath.endswith(".zip")
    if is_zipped:
        assert innerpath is not None, "Must specify innerpath for zip files."
    descriptor = {
        "name": make_valid_frictionless_name(resource_name),
        "type": "table",
        "path": filepath,
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
    return os.path.join(file.directory, f"{file.piece}.resource.json")

UTILS_DIR = os.path.dirname(__file__) # .../ms3/src/ms3/utils/
SCHEMAS_DIR = os.path.normpath(os.path.join(UTILS_DIR, '..', '..', '..', 'schemas'))
os.makedirs(SCHEMAS_DIR, exist_ok=True)


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
        index_levels = list(index_levels)
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
               include_index_levels: bool = False,
               ) -> dict | str:
    index_levels = df.index.names if include_index_levels else None
    result = get_schema_or_url(
        facet=facet,
        column_names=df.columns,
        index_levels=index_levels,)
    return result

@function_logger
def store_as_json_or_yaml(
        descriptor_dict: dict,
        descriptor_path: str):
    if descriptor_path.endswith(".yaml"):
        with open(descriptor_path, "w") as f:
            yaml.dump(descriptor_dict, f)
    elif descriptor_path.endswith(".json"):
        with open(descriptor_path, "w") as f:
            json.dump(descriptor_dict, f, indent=2)
    else:
        raise ValueError(
            f"Descriptor path must end with .yaml or .json: {descriptor_path}"
        )
    logger.info(f"Stored descriptor at {descriptor_path}.")


@function_logger
def make_resource_descriptor(
        df: pd.DataFrame,
        piece_name: str,
        facet: ScoreFacet,
        filepath: Optional[str] = None,
        innerpath: Optional[str] = None,
        include_index_levels: bool = False,
) -> dict:
    """ Given a dataframe and information about resource name and type and relative paths, create a frictionless
    resource descriptor.

    Args:
        df:
            Dataframe to be described. It is assumed (that is, blindly trusted) that it corresponds, or will correspond,
            to the file at ``filepath`` (or ``innerpath`` if zipped, see below).
        piece_name: Will be combined with facet to form the resource name.
        facet: Will be combined with piece_name to form the resource name. Specified separately because relevant for the schema.
        filepath:
            The relative path to the resource stored on disk, relative to the descriptor's location. Defaults to "<piece_name>.<facet>.tsv".
            Can be a path to a ZIP file, in which case the resource is stored in the ZIP file at ``innerpath``.
        innerpath:
            If ``filepath`` is a ZIP file, the resource is stored in the ZIP file at ``innerpath``. Defaults to "<piece_name>.<facet>.tsv".

    Returns:
        A frictionless resource descriptor dictionary.
    """
    schema = get_schema(
        df=df,
        facet=facet,
        include_index_levels=include_index_levels
    )
    resource_name = f"{piece_name}.{facet}"
    if filepath is None:
        filepath = f"{resource_name}.tsv"
    elif filepath.endswith(".zip"):
        if innerpath is None:
            innerpath = f"{resource_name}.tsv"
    return assemble_resource_descriptor(
        resource_name=resource_name,
        filepath=filepath,
        schema=schema,
        innerpath=innerpath,
        logger=logger
    )


@function_logger
def make_and_store_resource_descriptor(
        df: pd.DataFrame,
        directory: str,
        piece_name: str,
        facet: ScoreFacet,
        filepath: Optional[str] = None,
        innerpath: Optional[str] = None,
        descriptor_extension: Literal["json", "yaml"] = "json",
        include_index_levels: bool = False,
) -> str:
    """Make a resource descriptor for a given dataframe, store it to disk, and return the filepath.

    Args:
        df: Dataframe to be described.
        directory: Where to store the descriptor file.
        piece_name: Will be combined with facet to form the resource name.
        facet: Will be combined with piece_name to form the resource name. Specified separately because relevant for the schema.
        filepath:
            The relative path to the resource stored on disk, relative to the descriptor's location. Defaults to "<piece_name>.<facet>.tsv".
            Can be a path to a ZIP file, in which case the resource is stored in the ZIP file at ``innerpath``.
        innerpath:
            If ``filepath`` is a ZIP file, the resource is stored in the ZIP file at ``innerpath``. Defaults to "<piece_name>.<facet>.tsv".

    Returns:
        The path to the stored descriptor file.
    """
    descriptor_extension = descriptor_extension.lstrip(".")
    if descriptor_extension not in ("json", "yaml"):
        raise ValueError(f"Descriptor extension must be 'json' or 'yaml', not {descriptor_extension}")
    descriptor = make_resource_descriptor(
        df=df,
        piece_name=piece_name,
        facet=facet,
        filepath=filepath,
        innerpath=innerpath,
        include_index_levels=include_index_levels,
    )
    if descriptor['path'].endswith(".zip"):
        filepath =  descriptor['innerpath']
    else:
        filepath = descriptor['path']
    # because descriptor is named after the specific resource (not after the ZIP file which could be any collection of resources)
    descriptor_filepath = replace_extension(filepath, f'.resource.{descriptor_extension}')
    if directory:
        descriptor_path = os.path.join(directory, descriptor_filepath)
    else:
        descriptor_path = descriptor_filepath
    store_as_json_or_yaml(descriptor, descriptor_path, logger=logger)
    return descriptor_path

def validate_resource_descriptor(
        descriptor_path: str,
        raise_exception: bool = True,
) -> fl.Report:
    report = fl.validate(descriptor_path)
    if not report.valid:
        errors = [err.message for task in report.tasks for err in task.errors]
        if raise_exception:
            raise fl.FrictionlessException("\n".join(errors))
    return report


@function_logger
def make_and_store_and_validate_resource_descriptor(
        df: pd.DataFrame,
        directory: str,
        piece_name: str,
        facet: ScoreFacet,
        filepath: Optional[str] = None,
        innerpath: Optional[str] = None,
        descriptor_extension: Literal["json", "yaml"] = "json",
        include_index_levels: bool = False,
        raise_exception: bool = True,
) -> fl.Report:
    """Make a resource descriptor for a given dataframe, store it to disk, and return a validation report.

    Args:
        df: Dataframe to be described.
        directory: Where to store the descriptor file.
        piece_name: Will be combined with facet to form the resource name.
        facet: Will be combined with piece_name to form the resource name. Specified separately because relevant for the schema.
        filepath:
            The relative path to the resource stored on disk, relative to the descriptor's location. Defaults to "<piece_name>.<facet>.tsv".
            Can be a path to a ZIP file, in which case the resource is stored in the ZIP file at ``innerpath``.
        innerpath:
            If ``filepath`` is a ZIP file, the resource is stored in the ZIP file at ``innerpath``. Defaults to "<piece_name>.<facet>.tsv".
        include_index_levels:
            If False (default), the index levels are not described, assuming that they will not be written to disk
            (otherwise, validation error). Set to True to add all index levels to the described columns and, in addition,
            to make them the ``primaryKey`` (which, in frictionless, implies the constraints "required" & "unique"). In
            order to include the index levels as columns, but as primaryKey, simply pass ``df.reset_index()`` to the function.
        raise_exception: If True (default) raise if the resource is not valid.

    Returns:
        A frictionless validation report. ``report.valid`` returns a boolean that is True if successfully validated.
    """
    descriptor_path = make_and_store_resource_descriptor(
        df=df,
        directory=directory,
        facet=facet,
        piece_name=piece_name,
        filepath=filepath,
        innerpath=innerpath,
        descriptor_extension=descriptor_extension,
        include_index_levels=include_index_levels,
    )
    return validate_resource_descriptor(
        descriptor_path,
        raise_exception=raise_exception,
    )

@function_logger
def store_dataframe_resource(
        df: pd.DataFrame,
        directory: str,
        piece_name: str,
        facet: ScoreFacet,
        pre_process: bool = True,
        zipped: bool = False,
        frictionless: bool = True,
        descriptor_extension: Literal["json", "yaml", None] = "json",
        validate: bool = True,
        **kwargs):
    """Write a DataFrame to a TSV or CSV file together with its frictionless resource descriptor.
    If the resource comes with a single RangeIndex level, the index will be
    omitted from the TSV and the descriptor. If it comes with more than one level (a MultiIndex) the levels will be
    included as the left-most columns and declared as "primaryKey" in the descriptor.
    Uses: :py:func:`write_tsv`

    Args:
        df: DataFrame to write to disk and to store a descriptor for (if default ``frictionless=True``).
        directory: Where to write the file(s).
        piece_name: Name of the piece, used for the file name(s).
        facet: Name of the facet, used for the file name(s).
        pre_process:
            By default, DataFrame cells containing lists and tuples will be transformed to strings and Booleans will be
            converted to 0 and 1 (otherwise they will be written out as True and False). Pass False to prevent.
        zipped: If set to True, the TSV file will be written into a zip archive called ``<piece_name>.zip``.
        frictionless: If True (default), a frictionless resource descriptor will be written to disk as well.
        validate:
            If True (default), the frictionless resource descriptor will be validated against the schema, resulting
            in a FrictionlessException if the validation fails.
        **kwargs:
            Additional keyword arguments will be passed on to :py:meth:`pandas.DataFrame.to_csv`.
            Defaults arguments are ``index=False`` and ``sep='\t'`` (assuming extension '.tsv', see above) and,
            if ``zipped=True`` to the corresponding arguments.
    """
    tsv_name = f"{piece_name}.{facet}.tsv"
    if zipped:
        relative_filepath = f"{piece_name}.zip"
        innerpath = tsv_name
        read_csv_kwargs = dict(
            sep="\t",
            mode="a",
            compression=dict(method="zip", archive_name=tsv_name),
        )
        read_csv_kwargs.update(kwargs)
        msg = f"Written {tsv_name!r} to "
    else:
        relative_filepath = tsv_name
        innerpath = None
        read_csv_kwargs = dict(kwargs)
        msg = f"Written {facet!r} to "
    if directory:
        resource_path = os.path.join(directory, relative_filepath)
    else:
        resource_path = relative_filepath
    msg += resource_path
    if not isinstance(df.index, pd.RangeIndex):
        logger.debug(f"Keyword arguments for write_tsv() updated with index=True.")
        read_csv_kwargs["index"] = True
        include_index_levels = True
    else:
        include_index_levels = False
    write_tsv(
        df=df,
        file_path=resource_path,
        pre_process=pre_process,
        **read_csv_kwargs
    )
    logger.info(msg)
    if not frictionless:
        return resource_path
    descriptor_path = make_and_store_resource_descriptor(
        df=df,
        directory=directory,
        facet=facet,
        piece_name=piece_name,
        filepath=relative_filepath,
        innerpath=innerpath,
        descriptor_extension=descriptor_extension,
        include_index_levels=include_index_levels,
        logger=logger
    )
    if validate:
        validate_resource_descriptor(descriptor_path)
    return descriptor_path

@function_logger
def store_dataframes_package(
        dataframes: pd.DataFrame | Iterable[pd.DataFrame],
        facets: TSVtypes,
        directory: str,
        piece_name: str,
        pre_process: bool = True,
        zipped: bool = True,
        descriptor_extension: Literal["json", "yaml", None] = "json",
        validate: bool = True,
        **kwargs):
    """Write a DataFrame to a TSV or CSV file together with its frictionless resource descriptor.
    Uses: :py:func:`write_tsv`

    Args:
        dataframes: DataFrames to write into the same zip archive forming a datapackage.
        facets:
            Name of the facets, one per given dataframe. Appended to the file names of the TSV files in the form
            ``<piece_name>.<facet>.tsv``.
        directory: Where to create the ZIP file and its descriptor.
        piece_name: Name of the piece, used both for the names of ZIP file and the TSV files in includes.
        pre_process:
            By default, DataFrame cells containing lists and tuples will be transformed to strings and Booleans will be
            converted to 0 and 1 (otherwise they will be written out as True and False). Pass False to prevent.
        zipped: If set to False, the TSV file will not be written into a zip archive called ``<piece_name>.zip``.
        validate:
            If True (default), the frictionless resource descriptor will be validated against the schema, resulting
            in a FrictionlessException if the validation fails.
    """
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
    facets = resolve_facets_param(facets, TSVtype, none_means_all=False)
    package_descriptor = dict(
        name=piece_name,
        resources=[],
    )
    for df, facet in zip(dataframes, facets):
        resource_path = store_dataframe_resource(
            df=df,
            directory=directory,
            piece_name=piece_name,
            facet=facet,
            pre_process=pre_process,
            zipped=zipped,
            frictionless=False,
            descriptor_extension=descriptor_extension,
            validate=validate,
            logger=logger
        )
        directory, filepath = os.path.split(resource_path)
        resource_descriptor = make_resource_descriptor(
            df=df,
            piece_name=piece_name,
            facet=facet,
            filepath=filepath,
            include_index_levels=not isinstance(df.index, pd.RangeIndex),
            logger=logger,
        )
        package_descriptor["resources"].append(resource_descriptor)
    package_descriptor_filepath = f"{piece_name}.datapackage.{descriptor_extension}"
    package_descriptor_path = os.path.join(directory, package_descriptor_filepath)
    store_as_json_or_yaml(
        descriptor_dict=package_descriptor,
        descriptor_path=package_descriptor_path,
        logger=logger
    )
