### Spiking Neural Network algorithm </md>

In this folder all files are written similarily to the "pulses.py" explained [here](https://github.com/SamiNakouzi/YAMADA).

Here you will find a user's guide to running the algorithm:

**Objective of the algorithm:**

The objective of the algorithm is to implement a learning rule capable of finding a pump configuration that allows the laser to spike at a desired time for a given input.

**1. User's guide**
To start create a folder and name it **makegif** by writing this command on your console:
```python
mkdir makegif
```
The **"snn_test.py"** file contains the important parameters that the user can change. As stated above the script is written in similar fashion as the **pulses.py** file ([click here](https://github.com/SamiNakouzi/YAMADA) for guide).
The only parameter needed to be changed by the user is the input perturbation **eps_coh** which is a list of amplitudes for each perturbation. An example is already written:
```python
eps_coh= [0.03, 0.05, 0.03, 0.05, 0.03, 0.05, 0.03] 
```
all the other parameters don't need to be changed.

Then in the **"snn_model.py"** file there is the rule for the learning algorithm. the first important parameter is **t_d** which is a list representing the desired spike times. The user can define this list and an example that is already in the file is:
```python
td = [170, 350, 660]
```
Where we tell the algorith that we want our laser to spike at time **170**, **350**, and **660**.
The remaining parameters will be loaded from other text files and do not need any kinf of edditing.

In short to quickly use the algorithm the user will just have to change the input coherent perturbation **eps_coh** in the **snn_test.py** file and then define the desired output time **t_d** in the **snn_model.py** file.
After that to run the algorithm run on the terminal the **pipe.sh** file:
```
bash pipe.sh
```
The user will be then asked to chose over how many epochs they want to run the algorithm.

**2. Learning rule**

Consider we have an input spike train, an actual output spike train and a desired output spike train:

<img src="https://user-images.githubusercontent.com/60350687/181753201-9f9136cf-f92a-4926-8e08-1eaa78475351.png" alt="spike_train_example" width="350"/>
The learning rule that is going to tell the pump how to evolve under each input coherent perturbation is the following:

<img src="https://user-images.githubusercontent.com/60350687/181753026-fb7bdcab-3e5e-4b38-a0b0-5d170045a26e.png" alt="Learningrule" width="550"/>

