##################################################################
## This module is the main script to run the ABM-LU model. 
##################################################################

from landscape import build_landscape
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
### Making the graphs
####===============================================================####

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