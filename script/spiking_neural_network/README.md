### Inrodcution </md>

In this folder all files are written similarily to the "pulses.py" explained [here](https://github.com/SamiNakouzi/YAMADA).

Here you will find a user's guide to running the algorithm:

**Objective of the algorithm:**
The objective of the algorithm is to implement a learning rule capable of finding a pump configuration that allows the laser to spike at a desired time for a given input.

**User's guide**
The **"snn_test.py"** file contains the important parameters that the user can change. As stated above the script is written in similar fashion as the **pulses.py** file ([click here](https://github.com/SamiNakouzi/YAMADA) for guide).
The only parameter needed to be changed by the user is the input perturbation **eps_coh** which is a list of amplitudes for each perturbation. An example is already written:
```python
eps_coh= [0.03, 0.05, 0.03, 0.05, 0.03, 0.05, 0.03] 
```
all the other parameters don't need to be changed.

Then in the **"snn_model.py"** file there is the rule for the learning algorithm. the first important parameter is **t_d** which is a list representing the desired spike times.