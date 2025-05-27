# ChemReservoir
ChemReservoir: An Open-Source Framework for Chemically-Inspired Reservoir Computing

Copyright 2025 Mehmet Aziz Yirik

## Introduction

ChemReservoir is an open-source framework for chemically-inspired reservoir modelling. The tool allows the design of optimal chemically-inspired reservoir models based on genetic algorithms.
The combination of Gillespie algorithm based stochastic simulation and regression functions is performed for the evaluation of each individual in the genetic algorithm. The reservoir networks
are generated as cycle-based topologies with chords, local connections to evaluate the impact of the local neighborhood in information processing. 

## Scripts

The main ChemReservoir framework [script](https://github.com/MehmetAzizYirik/ChemReservoir/tree/main/scripts/chemReservoir.py) and all other related scripts are located in [the scripts folder](https://github.com/MehmetAzizYirik/ChemReservoir/tree/main/scripts). An example usage of ChemReservoir is given in [tests.py](https://github.com/MehmetAzizYirik/ChemReservoir/tree/main/scripts/tests.py).

## Dependencies

ChemReservoir utilizes the listed libraries:

- [MØD](https://cheminf.imada.sdu.dk/mod/)
- [deap](https://pypi.org/project/deap/)
- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)


## License
This project is licensed under the MIT License - see [the LICENSE file](https://github.com/MehmetAzizYirik/ChemReservoir/blob/main/LICENSE) file for details.

## Authors

 - Mehmet Aziz Yirik - [MehmetAzizYirik](https://github.com/MehmetAzizYirik) 
