"""Built-in plot generator implementations.

All modules in this package are imported eagerly so that their
``@plot_registry.register`` decorators fire at startup.
"""

from app.plots.builtin.boxplot import BoxPlot
from app.plots.builtin.ecdf import EcdfPlot
from app.plots.builtin.hierarchical import (
    ClusterMeanBarPlot,
    ClusterSpatialHeatmap,
    ProportionBarPlot,
)
from app.plots.builtin.violin import ViolinPlot

__all__ = [
    "BoxPlot",
    "EcdfPlot",
    "ViolinPlot",
    "ClusterMeanBarPlot",
    "ClusterSpatialHeatmap",
    "ProportionBarPlot",
]
