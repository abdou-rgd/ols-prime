"""Point-in-time features: a value at the close of bar t uses only bars final by then."""

from olsprime.features.basic import FEATURE_COLUMNS, basic_features

# The panel is loaded on first use: ``python -m olsprime.features.panel`` would otherwise import it twice,
# and the panel imports ``olsprime.labels``, which imports this package.
_PANEL_NAMES = ("PANEL_COLUMNS", "h1_subpanel_members", "hourly_panel", "member_months")

__all__ = ["FEATURE_COLUMNS", "basic_features", *_PANEL_NAMES]


def __getattr__(name: str):
    if name in _PANEL_NAMES:
        from olsprime.features import panel

        return getattr(panel, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
