from landscape import build_landscape

land = build_landscape(
    n_rows=20,
    n_cols=20,
    n_farmers=30,
    q_sigma=1.0,
    q_length_scale=2.5,
    farm_mu= 0.0,
    farm_sigma=1.0,
    seed=67,
)

### --------------------------------------------------------------------- ###
### Test the landscape module
print(land.summary())
plots = land.to_plot_dict()

qs = [p["q"] for p in plots.values()]
print(min(qs), max(qs))

owners = [p["owner"] for p in plots.values()]
print(all(o is not None for o in owners))

print(sum(len(v) for v in land.farmer_plot_ids.values()), land.n_plots)

### --------------------------------------------------------------------- ###