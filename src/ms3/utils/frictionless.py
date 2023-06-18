from functools import cache

import frictionless as fl
import pandas as pd

from ms3.utils import TSV_COLUMN_TITLES, TSV_COLUMN_DESCRIPTIONS, TSV_DTYPES, TSV_COLUMN_CONVERTERS, safe_frac, safe_int, str2inttuple, int2bool

FIELDS_WITHOUT_MISSING_VALUES = (
    'mc',
    'mc_playthrough',
)
FRACTION_REGEX = r"?\d+(?:\/\d+)?"
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


def make_frictionless_schema(df: pd.DataFrame,
                             as_descriptor: bool = True) -> dict | fl.Schema:
    fields = []
    for column_name in df.columns:
        field = column_name2frictionless_field(column_name)
        fields.append(field)
    descriptor = dict(fields=fields)
    if as_descriptor:
        return descriptor
    fl_schema = fl.Schema(descriptor)
    return fl_schema
