##################################################################
## This module is the main script to run the ABM-LU model. 
##################################################################

from landscape import build_landscape
from farm import build_farm_initializer, PortfolioRule
import parameters as p
import plots as pl

####===============================================================####
### Building the landscape
####===============================================================####

land = build_landscape(
    n_rows=p.n_rows,
    n_cols=p.n_cols,
    n_farmers=p.n_farmers,
    q_sigma=p.q_sigma,
    q_length_scale=p.q_length_scale,
    farm_mu=p.farm_mu,
    farm_sigma=p.farm_sigma,
    seed=p.seed,
)

### --------------------------------------------------------------------- ###
### Test the landscape module
print(land.summary())
# plots = land.to_plot_dict()

# qs = [p["q"] for p in plots.values()]
# print(min(qs), max(qs))

# owners = [p["owner"] for p in plots.values()]
# print(all(o is not None for o in owners))

# print(sum(len(v) for v in land.farmer_plot_ids.values()), land.n_plots)

### --------------------------------------------------------------------- ###


####===============================================================####
### Initializing the land use
####===============================================================####
farm_init = build_farm_initializer(seed=p.seed)

plots_by_id = land.to_plot_dict()
farmer_plot_ids = land.farmer_plot_ids

small_rules = [
    PortfolioRule(share_of_farms=0.50, share_I=1.00, share_O=0.00),  # 100% I
    PortfolioRule(share_of_farms=0.50, share_I=0.00, share_O=1.00),  # 100% O
]

medium_rules = [
    PortfolioRule(share_of_farms=0.30, share_I=1.00, share_O=0.00),  # 100% I
    PortfolioRule(share_of_farms=0.30, share_I=0.90, share_O=0.00),  # 90% I, 10% S
    PortfolioRule(share_of_farms=0.40, share_I=0.00, share_O=1.00),  # 100% O
]

large_rules = [
    PortfolioRule(share_of_farms=0.50, share_I=1.00, share_O=0.00),  # 100% I
    PortfolioRule(share_of_farms=0.50, share_I=0.90, share_O=0.00),  # 90% I, 10% S
]

initial_land_use = farm_init.initialize_land_use_by_farm_size(
    farmer_plot_ids=farmer_plot_ids,
    plots_by_id=plots_by_id,
    small_cutoff=0.50,
    large_cutoff=0.90,
    small_rules=small_rules,
    medium_rules=medium_rules,
    large_rules=large_rules,
    allocation_mode="quality_based",
)


####===============================================================####
### Making the graphs
####===============================================================####

## Initial plot quality and farm boundaries
pl.plot_quality_with_farm_borders(
    land,
    figsize=(8, 8),
    cmap="viridis",
    border_color="white",
    border_linewidth=1.0,
    show_axes=False,
    title="Initial plot quality and farm boundaries",
    savepath= p.savepath,
    show=False,
)

