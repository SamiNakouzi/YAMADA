### Inrodcution </md>


This is a project that aims at simulating a spiking laser using the Yamada model. It is easily possible to add coherent and incoherent perturbations to the system, and it is very easy to manipulate.
The Yamada model is the following:

$\dot{G} = b1 \left[\mu_1 - G(1+I) \right]$\
$\dot{Q} = b2 \left[ \mu_2 - Q(1+sI ) \right]$ \
$\dot{I} = I \left(G - Q - 1\right) - \beta_{sp} (1+\eta_1)^2$


**1. Running the model**

first import the model: **from sim import yamada\_model**
A simple line is used to run the model which is : **model = yamada\_model(s, mu1, mu2, eta1, beta, b1, b2, G0, Q0, I0)**
```python
from sim import yamada_model
model = yamada_model()

```

**2. Perturbations**

By default there are no perturbations added, however for the script to successfully run the perturbation line code should be written:\
**model.perturbate(t, pert_inc, pert_coh )**
all the arguments are optional and have default values. If no arguments are added there will be no perturbations. In the following each argument will be explained:

_NB: During this project perturbations were thought of as input bits. The script is written based on that representation._

- **t:** type: **array**. Are the timesteps desired. *default is: np.linspace(0, 2000, 2000)*

- **pert_inc:** type:**list** or **array**. Is a function defining incoherent perturbation. *default is 0* (no perturbation)

- **pert_coh:** type:**list** or **array**. Is a function defining coherent perturbation. *default is 0* (no perturbation)

Adding one or many perturbations requires the user to write a function that will shape the perturbation. An example is given in the "pulses.py" file where perturbations are moddeled as sqaure perturbation.
In the "script/pulses.py" file there are many parameters that the user can control to change the amplitude and duration of perturbations independently:

**eps_coh** and **eps_inc** are **lists** who's elements define the amplitude of each perturbation (index 0 for the first perturbation, index 1 for the second and so on...) for coherent and incoherent perturbations respectively.

**dt_coh** and **dt_inc** are **lists** who's elements define the duration of each perturbation independently.

**bit_coh** and **bit_inc** are **lists** who's elements define wether a perturbation is an input bit 1 or 0. This was only important for a specific project, but it has to be there for the code to run. However it is a parameter that can be ignored.

**pert_t_coh** and **pert_t_inc** are **lists** who's element define at what time each perturbation spikes independently.

Then perturbations are defined like the following:
```python
def perturbate_inc(t , dt, bits, pert_timing, neg_pulse = False):
    samples_t = t
    samples   = []
    perturbation = np.zeros((len(samples_t),))
    for idx, elem in enumerate(pert_timing):
        if bits[idx] == 1:
            perturbation[elem : elem + dt[idx]] = eps[idx]
            idx = idx +1
        elif bits[idx] == 0 and neg_pulse == True:
            perturbation[elem : elem + dt[idx]] = -eps[idx]
            idx = idx + 1
    samples = model.mu1 + perturbation
    
    samples_t, samples = np.array(samples_t), np.array(samples)
    return interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")
    
```
If **neg_pulse** is true, it will define negative perturbations if the bit = 0.

To run and integrate the user just needs to input the perturbation function in the perturbate function defining wether it is coherent or incoherent. In "pulses.py" we have both:

```python
pert_inc = perturbate_inc(t, dt_inc, eps_inc, bit_inc, pert_t_inc)
pert_coh = perturbate_coh(t, dt_coh, eps_coh, bit_coh, pert_t_coh)
model.perturbate(t, pert_inc, pert_coh)
model.integrate(t)
```
Finally to get the solution type:
```python
x = model.sol.tolist()
for idx in range(len(t)):
    intensity.append(x[idx][2])
    gain.append(x[idx][0])
    loss.append(x[idx][1])
```
Any type of perturbation can be defined, but will have to be written as time dependent interpolated functions.
