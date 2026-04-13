##################################################################
## This module defines the parameter values of the model.
##################################################################

#### Change the savepath variable to save the graph to a different location. ####
savepath = "Courses/S2-EEB/324-Theoretical-ecology/project-EEB324/ABM-LU-program/AMB-LU-graphs/initial_quality_map.png"

seed = 67

##### LANDSCAPE PARAMETERS #####

n_rows = 20
n_cols = 20
n_farmers = 30

farm_mu = 0.0 
farm_sigma = 1.0

q_sigma = 1.0
q_length_scale = 3.0

##### SIMULATION PARAMETERS #####
n_steps = 25 
regime = "policy" #choose from: "productivist", "market", "policy"

##### PRODUCTION PARAMETERS #####

output_price = 1.0

alpha_I = 1.2
organic_to_intensive_yield_ratio = 0.75
alpha_O = organic_to_intensive_yield_ratio * alpha_I
gamma_I = 1.0

##### COST PARAMETERS #####

c_I = 0.35
c_O = 0.28
c_S = 0.05

kappa_I = 0.30
kappa_O = 0.20
kappa_S = 0.02

##### FARMER HETEROGENEITY #####

eta_sigma = 0.10

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
