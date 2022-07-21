# ALChemE - _Assistive Libraries for Chemical Engineering_

ALChemE is a software to assist users in calculations related to the chemical engineering field and process design.
Currently, the software can assist users with the analysis of heat exchange networks (HENs) and water recycle networks (WReNs). Other relevant processes, such as distillation and reactors, may be implemented in the future.<br/><br/>
ALChemE can be operated via Terminal / IDE / Jupyter Notebooks or via a GUI. To learn how to use ALChemE via the former, please check the examples in [backend_examples](backend_examples). To learn how to use ALChemE via the latter, please check the [User Manual.pdf](User%20Manual.pdf).

## 1) Heat Exchanger Networks (HENs)
HEN analysis can assist users in maximizing energy recovery and minimizing costs.
Currently, users can input streams and utilities (with units of measurement), plot temperature interval diagrams and composite curves, and manually add heat exchangers.\
The software automatically handles unit conversions, calculates properties such as MER targets or the pinch point location, and can automatically place exchangers to minimize their number.
This function also returns the cost of the HENs generated.

## 2) Water Recycle Networks (WReNs)
WReN analysis can assist users in maximizing water recycling and minimizing costs.
Currently, users can input the flow rates and contaminant levels (with units of measurement) of water streams entering or leaving processes. Users can also define costs of individual matches (including those with freshwater or wastewater).\
The software automatically handles unit conversions and generates matches between processes to minimize water costs. Note the solution found may be a local minimum, so re-running the solver (or running it multiple times) may lead to a better solution. The best solution is always stored no matter how many times the solver is called.

