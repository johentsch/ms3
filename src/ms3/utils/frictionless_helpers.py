import hashlib
import json
import os
import re
from ast import literal_eval
from base64 import urlsafe_b64encode
from functools import cache
from pprint import pformat, pprint
from typing import Iterable, Literal, Optional, Tuple

import frictionless as fl
import pandas as pd
import yaml
from ms3._typing import ScoreFacet, TSVtype, TSVtypes
from ms3.logger import get_logger
from pandas.core.dtypes.common import is_integer_dtype

from .constants import (
    DEFAULT_CREATOR_METADATA,
    KEYSIG_DICT_ENTRY_REGEX,
    TIMESIG_DICT_ENTRY_REGEX,
)
from .functions import (
    TSV_COLUMN_CONVERTERS,
    TSV_COLUMN_DESCRIPTIONS,
    TSV_COLUMN_DTYPES,
    TSV_COLUMN_TITLES,
    File,
    eval_string_to_nested_list,
    replace_extension,
    resolve_facets_param,
    safe_frac,
    safe_int,
    str2inttuple,
    str2keysig_dict,
    str2timesig_dict,
    value2bool,
    write_tsv,
    write_validation_errors_to_file,
)

module_logger = get_logger(__name__)

FIELDS_WITHOUT_MISSING_VALUES = (
    "mc",
    "mc_playthrough",
)
FRACTION_REGEX = r"\d+(?:\/\d+)?"  # r"-?\d+(?:\/\d+)?" for including negative fractions
INT_ARRAY_REGEX = r"^[([]?(?:-?\d+\s*,?\s*)*[])]?$"  # allows any number of integers, separated by a comma and/or
# whitespace, and optionally enclosed in parentheses or square brackets
EDTF_LIKE_YEAR_REGEX = r"^\d{3,4}|\.{2}$"
# the following regexes match the strings produced when metadata2series() calls dict2oneliner() for TimeSig and KeySig
KEYSIG_DICT_REGEX = (
    f"^{{?({KEYSIG_DICT_ENTRY_REGEX})+}}?$"  # may or may not the outer curly braces,
)
TIMESIG_DICT_REGEX = f"^{{?({TIMESIG_DICT_ENTRY_REGEX})+}}?$"  # which need to be escaped in the f-strings


@cache
def column_name2frictionless_field(column_name) -> dict:
    global FRACTION_REGEX, INT_ARRAY_REGEX
    field = dict(
        name=column_name,
    )
    if column_name in FIELDS_WITHOUT_MISSING_VALUES:
        constraints = dict(required=True)
    else:
        constraints = dict()
    title = TSV_COLUMN_TITLES.get(column_name)
    description = TSV_COLUMN_DESCRIPTIONS.get(column_name)
    pandas_dtype = TSV_COLUMN_DTYPES.get(column_name, str)
    string_converter = TSV_COLUMN_CONVERTERS.get(column_name)
    if title:
        field["title"] = title
    if description:
        field["description"] = description
    if string_converter is not None:
        if string_converter == safe_frac:
            field["type"] = "string"
            constraints["pattern"] = FRACTION_REGEX
        elif string_converter == safe_int:
            if column_name in ("composed_start", "composed_end"):
                field["type"] = "string"
                constraints["pattern"] = EDTF_LIKE_YEAR_REGEX
            else:
                field["type"] = "integer"
                field[
                    "bareNumber"
                ] = False  # allow other leading and trailing characters
        elif string_converter == str2inttuple:
            field["type"] = "string"
            constraints["pattern"] = INT_ARRAY_REGEX
        elif string_converter == value2bool:
            field["type"] = "boolean"
        elif string_converter in (literal_eval, eval_string_to_nested_list):
            field["type"] = "array"
        elif string_converter == str2keysig_dict:
            field["type"] = "string"
            constraints["pattern"] = KEYSIG_DICT_REGEX
        elif string_converter == str2timesig_dict:
            field["type"] = "string"
            constraints["pattern"] = TIMESIG_DICT_REGEX
        else:
            raise NotImplementedError(
                f"Unfamiliar with string converter {string_converter}"
            )
    elif pandas_dtype:
        if pandas_dtype in (int, "Int64"):
            field["type"] = "integer"
        elif pandas_dtype == float:
            field["type"] = "number"
        elif pandas_dtype in (str, "string"):
            field["type"] = "string"
        else:
            raise NotImplementedError(
                f"Don't know how to handle pandas dtype {pandas_dtype}"
            )
    else:
        raise NotImplementedError(f"Don't know how to handle column {column_name}")
    if len(constraints) > 0:
        field["constraints"] = constraints
    return field


def make_frictionless_schema_descriptor(
    column_names: Iterable[str],
    primary_key: Optional[Iterable[str]] = None,
    **custom_data,
) -> dict:
    fields = []
    if primary_key:
        for ix_level, column in zip(primary_key, column_names):
            if ix_level != column:
                raise ValueError(
                    f"primary_key {primary_key} does not match column_names {column_names[:len(primary_key)]}"
                )
    for column_name in column_names:
        field = column_name2frictionless_field(column_name)
        if "type" not in field:
            raise ValueError(
                f"column_name2frictionless_field({column_name!r}) = {field} (missing 'type'!)"
            )
        fields.append(field)
    descriptor = dict(fields=fields)
    if primary_key:
        descriptor["primaryKey"] = list(primary_key)
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


def assemble_resource_descriptor(
    resource_name: str,
    filepath: str,
    schema: str | dict,
    innerpath: Optional[str] = None,
    **kwargs,
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
    descriptor.update(
        {"encoding": "utf-8", "dialect": {"csv": {"delimiter": "\t"}}, "schema": schema}
    )
    if kwargs:
        descriptor.update(kwargs)
    return descriptor


def make_json_path(file: File) -> str:
    return os.path.join(file.directory, f"{file.piece}.resource.json")


UTILS_DIR = os.path.dirname(__file__)  # .../ms3/src/ms3/utils/
SCHEMAS_DIR = os.path.normpath(
    os.path.join(UTILS_DIR, "..", "..", "..", "frictionless_schemas")
)
os.makedirs(SCHEMAS_DIR, exist_ok=True)


def get_truncated_hash(
    S: str | Iterable[str], hash_func=hashlib.sha1, length=10
) -> str:
    """Computes the given hashfunction for the given string(s), and truncates the result.

    Raises:
        ValueError: If the hash function cannot be computed for any of the strings in S.
    """
    if isinstance(S, str):
        S = [S]
    hasher = hash_func()
    for s in S:
        try:
            hasher.update(s.encode("utf-8"))
        except AttributeError as e:
            raise ValueError(f"Element {s!r} from {S} resulted in error {e!r}") from e
    return urlsafe_b64encode(hasher.digest()[:length]).decode("utf-8").rstrip("=")


def get_schema_or_url(
    facet: str,
    column_names: Tuple[str],
    index_levels: Optional[Tuple[str]] = None,
    base_local_path=SCHEMAS_DIR,
    base_url="https://raw.githubusercontent.com/DCMLab/frictionless_schemas/main/",
    **kwargs
    # "https://raw.githubusercontent.com/johentsch/ms3/main/schemas/"
) -> str | dict:
    """Given a facet name (=subfolder) and a tuple of [index column names +] column names, compute an identifier and
    if that schema exists under ``<base_url>/<facet>/<identifier>.schema.yaml`` return that URL, or otherwise
    create a the frictionless schema descriptor based on the column names, using the descriptions and types the ms3
    stores for known column names (see :func:`make_frictionless_schema_descriptor`) and treating unknown columns
    as string fields. In the latter case, the YAML file is written to
    ``<base_local_path>/<facet>/<identifier>.schema.yaml``
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
            unless the schema is found online. The purpose of this is to allow for easy updating of the online
            repository by setting this argument to a local clone. The default value is ``<ms3>/frictionless_schemas/``,
            a submodule (if initialized) corresponding to https://github.com/DCMLab/frictionless_schemas.
        base_url:
            If schema descriptor is found at``<base_url>/<facet>/<identifier>.schema.yaml``, the function returns the
            URL rather than the descriptor dict.
        **kwargs:
            Arbitrary key-value pairs that will be added to the frictionless schema descriptor as "custom" metadata.

    Returns:

    """
    if base_url[-1] != "/":
        base_url += "/"
    if index_levels is None:
        column_names = tuple(column_names)
    else:
        column_names = tuple(index_levels) + tuple(column_names)
        index_levels = list(index_levels)
    schema_identifier = get_truncated_hash(column_names)
    schema_filename = f"{schema_identifier}.schema.yaml"
    schema_filepath = f"{facet}/{schema_filename}"  # for URL & uniform filepath
    schema_path = os.path.join(base_local_path, facet, schema_filename)  # for local OS
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
                descriptor = json.load(open(schema_path, "r"))
                pprint(descriptor)
                raise
    else:
        descriptor = make_frictionless_schema_descriptor(
            column_names=column_names,
            primary_key=index_levels,
            # the rest is custom data added to the schema descriptor
            facet=facet,
            identifier=schema_identifier,
            filepath=schema_filepath,
            **kwargs,
        )
        fl.Schema(descriptor).to_yaml(schema_path)
        return descriptor


def get_schema(
    df: pd.DataFrame,
    facet: str,
    include_index_levels: bool = False,
    base_local_path=SCHEMAS_DIR,
    base_url="https://raw.githubusercontent.com/DCMLab/frictionless_schemas/main/",
    **kwargs,
) -> dict | str:
    """Given a dataframe and a facet name, return a frictionless schema descriptor for the dataframe.
    If the schema with the exact same sequence of columns (and index levels) is accessible online at
    ``base_url/facet/<identifier>.schema.yaml``, return that URL, otherwise return the descriptor itself as a dict.
    In both cases, the schema is stored at ``base_local_path/facet/<identifier>.schema.yaml`` if it does not exist.

    Args:
        df: Dataframe to create a schema for.
        facet: Facet that the dataframe describes, used as subfolder and added as custom metadata to the schema.
        include_index_levels:
            If False (default), the index levels are not described, assuming that they will not be written to disk
            (otherwise, validation error). Set to True to add all index levels to the described columns and,
            in addition, to make them the ``primaryKey`` (which, in frictionless, implies the constraints "required" &
            "unique").
        base_local_path:
            Schema descriptors will be created locally under ``<base_local_path>/<facet>/<identifier>.schema.yaml``,
            unless the schema is found online. The purpose of this is to allow for easy updating of the online
            repository by setting this argument to a local clone. The default value is ``<ms3>/frictionless_schemas/``,
            a submodule (if initialized) corresponding to https://github.com/DCMLab/frictionless_schemas.
        base_url:
            If schema descriptor is found at``<base_url>/<facet>/<identifier>.schema.yaml``, the function returns the
            URL rather than the descriptor dict.
        **kwargs:
            Arbitrary key-value pairs that will be added to the frictionless schema descriptor as "custom" metadata.

    Returns:

    """
    index_levels = df.index.names if include_index_levels else None
    result = get_schema_or_url(
        facet=facet,
        column_names=df.columns,
        index_levels=index_levels,
        base_local_path=base_local_path,
        base_url=base_url,
        **kwargs,
    )
    return result


def store_as_json_or_yaml(descriptor_dict: dict, descriptor_path: str, logger=None):
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
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


def make_resource_descriptor(
    df: pd.DataFrame,
    piece_name: str,
    facet: str,
    filepath: Optional[str] = None,
    innerpath: Optional[str] = None,
    include_index_levels: bool = False,
    **kwargs,
) -> dict:
    """Given a dataframe and information about resource name and type and relative paths, create a frictionless
    resource descriptor.

    Args:
        df:
            Dataframe to be described. It is assumed (that is, blindly trusted) that it corresponds, or will correspond,
            to the file at ``filepath`` (or ``innerpath`` if zipped, see below).
        piece_name: Will be combined with facet to form the resource name.
        facet: Will be combined with piece_name to form the resource name. Specified separately because relevant for
        the schema.
        filepath:
            The relative path to the resource stored on disk, relative to the descriptor's location. Defaults to
            "<piece_name>.<facet>.tsv".
            Can be a path to a ZIP file, in which case the resource is stored in the ZIP file at ``innerpath``.
        innerpath:
            If ``filepath`` is a ZIP file, the resource is stored in the ZIP file at ``innerpath``. Defaults to
            "<piece_name>.<facet>.tsv".
        **kwargs
            Additional keyword arguments written as metadata into the descriptor.
    Returns:
        A frictionless resource descriptor dictionary.
    """
    schema = get_schema(
        df=df,
        facet=facet,
        include_index_levels=include_index_levels,
        used_in=piece_name,
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
        **kwargs,
    )


def make_and_store_resource_descriptor(
    df: pd.DataFrame,
    directory: str,
    piece_name: str,
    facet: str,
    filepath: Optional[str] = None,
    innerpath: Optional[str] = None,
    descriptor_extension: Literal["json", "yaml"] = "json",
    include_index_levels: bool = False,
    logger=None,
    **kwargs,
) -> str:
    """Make a resource descriptor for a given dataframe, store it to disk, and return the filepath.

    Args:
        df: Dataframe to be described.
        directory: Where to store the descriptor file.
        piece_name: Will be combined with facet to form the resource name.
        facet: Will be combined with piece_name to form the resource name. Specified separately because relevant for
        the schema.
        filepath:
            The relative path to the resource stored on disk, relative to the descriptor's location. Defaults to
            "<piece_name>.<facet>.tsv".
            Can be a path to a ZIP file, in which case the resource is stored in the ZIP file at ``innerpath``.
        innerpath:
            If ``filepath`` is a ZIP file, the resource is stored in the ZIP file at ``innerpath``. Defaults to
            "<piece_name>.<facet>.tsv".
        **kwargs
            Additional keyword arguments written as metadata into the descriptor.
    Returns:
        The path to the stored descriptor file.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    descriptor_extension = descriptor_extension.lstrip(".")
    if descriptor_extension not in ("json", "yaml"):
        raise ValueError(
            f"Descriptor extension must be 'json' or 'yaml', not {descriptor_extension}"
        )
    descriptor = make_resource_descriptor(
        df=df,
        piece_name=piece_name,
        facet=facet,
        filepath=filepath,
        innerpath=innerpath,
        include_index_levels=include_index_levels,
        **kwargs,
    )
    if descriptor["path"].endswith(".zip"):
        filepath = descriptor["innerpath"]
    else:
        filepath = descriptor["path"]
    # because descriptor is named after the specific resource (not after the ZIP file which could be any collection
    # of resources)
    descriptor_filepath = replace_extension(
        filepath, f".resource.{descriptor_extension}"
    )
    if directory:
        descriptor_path = os.path.join(directory, descriptor_filepath)
    else:
        descriptor_path = descriptor_filepath
    store_as_json_or_yaml(descriptor, descriptor_path, logger=logger)
    return descriptor_path


def validate_descriptor_at_path(
    descriptor_path: str,
    raise_exception: bool = True,
    write_or_remove_errors_file: bool = True,
    logger=None,
) -> fl.Report:
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    report = fl.validate(descriptor_path)
    validation_tasks = []
    if report.valid:
        logger.debug(f"{descriptor_path} successfully validated.")
    else:
        validation_tasks, error_lists = zip(
            *(
                (pformat(task), [err.message for err in task.errors])
                for task in report.tasks
            )
        )
        all_errors = sum(error_lists, [])
        errors_block = "\n".join(all_errors)
        logger.warning(
            f"Validation of {descriptor_path} failed with {len(all_errors)} validation errors:\n{errors_block}",
            extra={"message_id": (32,)},
        )
    if write_or_remove_errors_file:
        # the following call removes the .errors file if the validation was successful
        errors_file = replace_extension(descriptor_path, ".errors")
        header = (
            f"To reproduce: frictionless validate {os.path.basename(descriptor_path)}"
        )
        write_validation_errors_to_file(
            errors_file=errors_file,
            errors=validation_tasks,
            header=header,
            logger=logger,
        )
    if not report.valid and raise_exception:
        raise fl.FrictionlessException("\n".join(all_errors))
    return report


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
    write_or_remove_errors_file: bool = True,
    logger=None,
    **kwargs,
) -> fl.Report:
    """Make a resource descriptor for a given dataframe, store it to disk, and return a validation report.

    Args:
        df: Dataframe to be described.
        directory: Where to store the descriptor file.
        piece_name: Will be combined with facet to form the resource name.
        facet: Will be combined with piece_name to form the resource name. Specified separately because relevant for
        the schema.
        filepath:
            The relative path to the resource stored on disk, relative to the descriptor's location. Defaults to
            "<piece_name>.<facet>.tsv".
            Can be a path to a ZIP file, in which case the resource is stored in the ZIP file at ``innerpath``.
        innerpath:
            If ``filepath`` is a ZIP file, the resource is stored in the ZIP file at ``innerpath``. Defaults to
            "<piece_name>.<facet>.tsv".
        include_index_levels:
            If False (default), the index levels are not described, assuming that they will not be written to disk
            (otherwise, validation error). Set to True to add all index levels to the described columns and,
            in addition,
            to make them the ``primaryKey`` (which, in frictionless, implies the constraints "required" & "unique"). In
            order to include the index levels as columns, but as primaryKey, simply pass ``df.reset_index()`` to the
            function.
        raise_exception:  If True (default) raise if the resource is not valid. Only relevant when frictionless=True
        (i.e., by default).
        write_or_remove_errors_file:
            If True (default) write a .errors file if the resource is not valid, otherwise remove it if it exists.
            Only relevant when frictionless=True (i.e., by default).
        **kwargs
            Additional keyword arguments written as metadata into the descriptor.

    Returns:
        A frictionless validation report. ``report.valid`` returns a boolean that is True if successfully validated.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
    descriptor_path = make_and_store_resource_descriptor(
        df=df,
        directory=directory,
        facet=facet,
        piece_name=piece_name,
        filepath=filepath,
        innerpath=innerpath,
        descriptor_extension=descriptor_extension,
        include_index_levels=include_index_levels,
        logger=logger,
        **kwargs,
    )
    return validate_descriptor_at_path(
        descriptor_path,
        raise_exception=raise_exception,
        write_or_remove_errors_file=write_or_remove_errors_file,
        logger=logger,
    )


def is_range_index_equivalent(idx: pd.Index) -> bool:
    """Check if a given index is a RangeIndex with the same start, stop, and step as the default RangeIndex."""
    if isinstance(idx, pd.RangeIndex):
        return True
    if is_integer_dtype(idx.dtype) and idx.is_monotonic_increasing and idx[0] == 0:
        return True
    return False


def all_index_levels_named(idx: pd.Index) -> bool:
    if any(not level_name for level_name in idx.names):
        return False
    return True


def store_dataframe_resource(
    df: pd.DataFrame,
    directory: str,
    piece_name: str,
    facet: str,
    pre_process: bool = True,
    zipped: bool = False,
    frictionless: bool = True,
    descriptor_extension: Literal["json", "yaml", None] = "json",
    raise_exception: bool = True,
    write_or_remove_errors_file: bool = True,
    logger=None,
    custom_metadata: dict = None,
    **kwargs,
) -> Optional[str]:
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
        raise_exception:  If True (default) raise if the resource is not valid. Only relevant when frictionless=True
        (i.e., by default).
        write_or_remove_errors_file:
            If True (default) write a .errors file if the resource is not valid, otherwise remove it if it exists.
            Only relevant when frictionless=True (i.e., by default).

        **kwargs:
            Additional keyword arguments will be passed on to :py:meth:`pandas.DataFrame.to_csv`.
            Defaults arguments are ``index=False`` and ``sep='\t'`` (assuming extension '.tsv', see above) and,
            if ``zipped=True`` to the corresponding arguments.

    Returns:
        If ``frictionless=False``, the path to the written resource.
        If ``frictionless=True``, the path to the written descriptor or None if it could not be generated.
    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
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
    if not all_index_levels_named(df.index) or is_range_index_equivalent(df.index):
        include_index_levels = False
    else:
        logger.debug("Keyword arguments for write_tsv() updated with index=True.")
        read_csv_kwargs["index"] = True
        include_index_levels = True
    write_tsv(
        df=df, file_path=resource_path, pre_process=pre_process, **read_csv_kwargs
    )
    logger.info(msg)
    if not frictionless:
        return resource_path
    try:
        if custom_metadata is None:
            custom_metadata = {}
        descriptor_path = make_and_store_resource_descriptor(
            df=df,
            directory=directory,
            facet=facet,
            piece_name=piece_name,
            filepath=relative_filepath,
            innerpath=innerpath,
            descriptor_extension=descriptor_extension,
            include_index_levels=include_index_levels,
            creator=DEFAULT_CREATOR_METADATA,  # custom metadata field for descriptor, passed as kwarg
            logger=logger,
            **custom_metadata,
        )
    except ValueError as e:
        descriptor_path = None
        logger.warning(
            f"Could not create frictionless descriptor for {resource_path} due to this error: {e}",
        )
    if descriptor_path is not None:
        validate_descriptor_at_path(
            descriptor_path,
            raise_exception=raise_exception,
            write_or_remove_errors_file=write_or_remove_errors_file,
            logger=logger,
        )
    return descriptor_path


def store_dataframes_package(
    dataframes: pd.DataFrame | Iterable[pd.DataFrame],
    facets: TSVtypes,
    directory: str,
    piece_name: str,
    pre_process: bool = True,
    zipped: bool = True,
    frictionless: bool = True,
    descriptor_extension: Literal["json", "yaml", None] = "json",
    raise_exception: bool = True,
    write_or_remove_errors_file: bool = True,
    logger=None,
    custom_metadata: Optional[dict] = None,
):
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
        frictionless:
            If True (default), the package is written together with a frictionless package descriptor JSON/YAML file
            that includes column schemas of the included TSV files which are used to validate them all at once.
        raise_exception:  If True (default) raise if the resource is not valid. Only relevant when frictionless=True
        (i.e., by default).
        write_or_remove_errors_file:
            If True (default) write a .errors file if the resource is not valid, otherwise remove it if it exists.
            Only relevant when frictionless=True (i.e., by default).

    """
    if logger is None:
        logger = module_logger
    elif isinstance(logger, str):
        logger = get_logger(logger)
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
            logger=logger,
        )
        directory, filepath = os.path.split(resource_path)
        resource_descriptor = make_resource_descriptor(
            df=df,
            piece_name=piece_name,
            facet=facet,
            filepath=filepath,
            include_index_levels=not isinstance(df.index, pd.RangeIndex),
        )
        package_descriptor["resources"].append(resource_descriptor)
    if not frictionless:
        return
    package_descriptor["creator"] = DEFAULT_CREATOR_METADATA  # custom metadata field
    if custom_metadata is not None:
        package_descriptor.update(custom_metadata)
    package_descriptor_filepath = f"{piece_name}.datapackage.{descriptor_extension}"
    package_descriptor_path = os.path.join(directory, package_descriptor_filepath)
    store_as_json_or_yaml(
        descriptor_dict=package_descriptor,
        descriptor_path=package_descriptor_path,
        logger=logger,
    )
    _ = validate_descriptor_at_path(
        package_descriptor_path,
        raise_exception=raise_exception,
        write_or_remove_errors_file=write_or_remove_errors_file,
        logger=logger,
    )
