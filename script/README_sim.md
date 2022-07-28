### Inrodcution </md>

The "sim.py" file is a class of the model for spiking lasers. It uses the [odint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html) package to solve the Yamada equations.

The parameters are predefined and can be changed.

The **perturbate** function's role is to add perturbations. This function has to be called in any script file wether or not perturbations are added. If no perturbations are added then the **perturbate** function will only apply the predefined parameters as interpolated time dependent functions to the Yamada equations. If perturbations are defined and called, the **perturbate** function will add those perturbations. 

Perturbations have to be defined as interpolated time dependent functions.
