#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pybamm
import matplotlib.pyplot as plt
import numpy as np
model_dfn = pybamm.lithium_ion.DFN()
sim_dfn = pybamm.Simulation(model_dfn)
sim_dfn.solve([0, 3600])


# In[28]:


import pybamm
import matplotlib.pyplot as plt
import numpy as np
#Initialise model
model = pybamm.BaseModel()
#Define parameters and variables
c = pybamm.Variable ( "c" , domain = "unit line" )
k = pybamm.Parameter ( "Diffusion parameter" )
#State governing equations
D = k * (1 + c )
dcdt = pybamm.div ( D * pybamm . grad ( c ) )
model . rhs = { c : dcdt }
#state boundary conditions
D_right = pybamm.BoundaryValue (D , "right" )
model.boundary_conditions = {
    c: {
     " left": (1 , "Dirichlet" ) ,
     " right": (1/ D_right, "Neumann" )
    }
}
# State initial conditions
x = pybamm . SpatialVariable ("x", domain = "unit line")
model . initial_conditions = {c : x + 1} 

#Load model and geometry
model = pybamm.lithium_ion.DFN()
geometry = model.default_geometry 
#Process parameters
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)
#Set mesh
mesh = pybamm.Mesh(
    geometry,
    model.default_submesh_types,
    model.default_var_pts)
#Discretise model
disc = pybamm.Discretisation(
         mesh, model.default_spatial_methods
       )
disc.process_model(model)
#Solve model
t_eval = np.linspace(0, 0.2, 100)
solution = model.default_solver.solve(model, t_eval) 
models = [
    pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.DFN(),
]
sims = []
for model in models:
    sim = pybamm.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)
pybamm.dynamic_plot(sims, time_unit="seconds")


# In[ ]:




