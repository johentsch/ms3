import json

from ms3.utils.frictionless_helpers import get_schema_or_url


def test_schema_descriptor_order_is_stable(tmp_path):
    """A freshly created schema descriptor must serialize identically to the one loaded from the stored YAML."""
    columns = ("mc", "mn", "quarterbeats", "label")
    kwargs = dict(
        facet="harmonies",
        column_names=columns,
        base_local_path=str(tmp_path),
        base_url="https://example.invalid/",
        used_in="piece",
    )
    created = get_schema_or_url(**kwargs)
    loaded = get_schema_or_url(**kwargs)
    assert json.dumps(created, indent=2) == json.dumps(loaded, indent=2)
