import uncertainpy as un
import chaospy as cp
import h5py

from air import Icestupa

# Define a parameter list
ftl_dist = cp.Uniform(0, 1)

parameters = {"ftl": ftl_dist}
parameters = un.Parameters(parameters)

# Initialize the model
model = Icestupa()

# Create a model from the coffee_cup function and add labels
model = un.Model(run=model.melt_freeze, labels=["IceV ", "Efficiency"])

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
data = UQ.quantify()

print(data["melt_freeze"])

plt.plot(data["melt_freeze"].time, data["melt_freeze"].mean)
plt.xlabel(data["melt_freeze"].labels[0])
plt.ylabel(data["melt_freeze"].labels[1])
plt.show()
