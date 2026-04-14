##################################################################
## This module defines the parameter values of the model.
##################################################################

#### Change the savepath variable to save the graph to a different location. ####
savepath = "Courses/S2-EEB/324-Theoretical-ecology/project-EEB324/ABM-LU-program/AMB-LU-graphs/initial_quality_map.png"

seed = 67

##### LANDSCAPE PARAMETERS #####

n_rows = 10
n_cols = 10
n_farmers = 10

farm_mu = 0.0 
farm_sigma = 1.0

q_sigma = 1.0
q_length_scale = 3.0

##### SIMULATION PARAMETERS #####
n_steps = 25 
regime = "policy"           #choose from: "productivist", "market", "policy"
subsidy_type = "practice"   #choose from: "practice", "conversion", "results"

##### PRODUCTION PARAMETERS #####

output_price = 1.0

alpha_I = 1.2

organic_to_intensive_yield_ratio = 0.75
alpha_O = organic_to_intensive_yield_ratio * alpha_I

gamma_I = 1.0
gamma_0 = 1.0

##### PRODUCTION COST PARAMETERS #####

# Higher fixed-cost and higher sensitivity to land quality for intensive agriculture compared to organic
c_I = 0.35
kappa_I = 0.30

c_O = 0.28
kappa_O = 0.20

# no production cost for set-aside land
c_S = 0.0
kappa_S = 0.0

##### FARMER HETEROGENEITY #####

sigma_eta_I = 0.010
sigma_eta_O = 0.015

##### ENVIRONMENTAL DYNAMICS #####

r_S = 0.06
r_O = 0.03
d_I = 0.05
theta = 0.02
initial_E = 0.0

##### PRACTICE-BASED SUBSIDY #####

s_O = 0.10
s_S = 0.22

##### CONVERSION SUBSIDY #####

s_C = 0.25

##### RESULTS-BASED SUBSIDY #####

beta = 0.15
