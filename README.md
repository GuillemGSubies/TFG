# TFG
Code for my Mathematics and Computer Science degrees thesis

## Index
* [Prerrequisites](#prerrequisites)
* [Installation](#installation)

## Prerrequisites
* This code has been done in a machine running Ubuntu and it is meant to run in Ubuntu. It may or may not work in other operating systems.
* A Python3 Anaconda should be installed: [Guide](https://docs.anaconda.com/anaconda/install/linux/)

## Installation

### CPU Version
The first time you execute the code type in a terminal:

```bash
conda env create -f environment.yml
```

In order to activate the environment type:

```bash
source activate tfg_guillem
```

When you are done:

```bash
source deactivate
```

### GPU Version

This version has the same steps. You just have to add `_gpu` after the env file name when creating and after the env name when activating.



### Citation

If you are citing the theorical work and the state-of-the-art:

```
@misc{subies_2019_mat,
  title={`Estudio teórico sobre modelos de secuencias con redes neuronales recurrentes para la generación de texto},
  url={https://github.com/GuillemGSubies/TFG},
  author={Subies, Guillem García},
  year={2019}
} 
```

If you want to cite the code:

```
@misc{subies_2019_inf,
  title={`Estudio práctico sobre modelos de secuencias con redes neuronales recurrentes para la generación de texto},
  url={https://github.com/GuillemGSubies/TFG},
  author={Subies, Guillem García},
  year={2019}
} 
```
