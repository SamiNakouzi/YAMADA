### Inrodcution </md>


This is a project that aims at simulating a spiking laser using the Yamada model. It is easily possible to add coherent and incoherent perturbations to the system, and it is very easy to manipulate.
The Yamada model is the following:

$\left\{ \dot{G} = b1 \left[\mu_1 - G(1+I) \right]$ \\ 
$\dot{Q} = b2 \left[ \mu_2 - Q(1+sI ) \right]$ \\
$dot{I} = I \left(G - Q - 1) - \beta_{sp} (1+\eta_1)^2$\\


**1. Running the model**

first import the model: *from sim import yamada\_model*
A simple line is used to run the model which is : *model = yamada\_model(s, mu1, mu2, eta1, beta, b1, b2, G0, Q0, I0)*
![Import](img/import.png)
**2. Perturbations**
By default there are no perturbations added, however for the script to successfully run the perturbation line code should be written:
*model.perturbate(t, dt, eps, bits, pert\_timing, neg\_pulse )*
all the arguments are optional and have default values so that woth no arguments there are no perturbations. In the following each argument will be explained:

- **t:** type: **array**. Are the timesteps desired. *default is: np.linspace(0, 2000, 2000)*

- **dt:** type:**int** or **array**. Is the duration of the perturbation. *default is 0*

- **eps:** type **int**. Is the perturbation amplitude. *default is 0*

- **bits:** type **array**. This functionality allows the user to convert bits intp perturbations. if the given bit is 1, then we will have a perturbation, if it is zero we will
we won't add a perturbation. *As a default value it is set as [1] so that if the user wants a perturbation without thinking about how the code is written, then a perturbation will be added*
 
- **pert__\__timing:** Type **array**. Is the time(s) at which the user wishes to add a perturbation. *Default value is [300].*

- **neg__\__pulse:** If **True** will execute a negative perturbation if the bit is zero. *Default value is ****false****.

\end{document}
