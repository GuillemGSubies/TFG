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
