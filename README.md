### Inrodcution </md>


This is a project that aims at simulating a spiking laser using the Yamada model. It is easily possible to add coherent and incoherent perturbations to the system, and it is very easy to manipulate.
The Yamada model is the following:

$\dot{G} = b1 \left[\mu_1 - G(1+I) \right]$\
$\dot{Q} = b2 \left[ \mu_2 - Q(1+sI ) \right]$ \
$\dot{I} = I \left(G - Q - 1\right) - \beta_{sp} (1+\eta_1)^2$


**1. Running the model**

first import the model: **from sim import yamada\_model**\
A simple line is used to run the model which is : **model = yamada\_model(s, mu1, mu2, eta1, beta, b1, b2, G0, Q0, I0)**\
![import](https://user-images.githubusercontent.com/60350687/180771050-1703a062-35d6-4f6c-8ae2-44fe29b3d713.png)
\
**2. Perturbations**\

By default there are no perturbations added, however for the script to successfully run the perturbation line code should be written:\
**model.perturbate(t, pert_inc, pert_coh )**\
all the arguments are optional and have default values. If no arguments are added there will be no perturbations. In the following each argument will be explained:

_NB: During this project perturbations were thought of as input bits. The script is written based on that representation._

- **t:** type: **array**. Are the timesteps desired. *default is: np.linspace(0, 2000, 2000)*

- **pert_inc:** type:**list** or **array**. Is a function defining incoherent perturbation. *default is 0* (no perturbation)

- **pert_coh:** type:**list** or **array**. Is a function defining coherent perturbation. *default is 0* (no perturbation)

Adding one or many perturbations requires the user to write a function that will shape the perturbation. An example is given in the "pulses.py" file where perturbations are moddeled as sqaure perturbation.
In the "pulses.py" file there are many parameters that the user can control to change the amplitude and duration of perturbations independently:
**eps_\_coh** and **eps_\_inc** are **lists** who's elements define the amplitude of each perturbation (index 0 for the first perturbation, index 1 for the second and so on...) for coherent and incoherent perturbations respectively.

**dt_\_coh** and **dt_\_inc** are **lists** who's elements define the duration of each perturbation independently.

**bit_\_coh** and **bit_\_inc** are **lists** who's elements define wether a perturbation is an input bit 1 or 0. This was only important for a specific project, but it has to be there for the code to run. However it is a parameter that can be ignored.

**pert_\_t_\_coh** and **pert_\_t_\_inc** are **lists** who's element define at what time each perturbation spikes independently.

Then the perturbation function is defined as the following example:
![perturbation example](https://user-images.githubusercontent.com/60350687/181457655-66346cc5-1f27-48ab-adc2-b5fb2c315fbf.png)
If **neg_pulse** is true, it will define negative perturbations if the bit = 0.
\end{document}
