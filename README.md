# ALChemE - _Assistive Libraries for Chemical Engineering_

ALChemE is a software to assist users in calculations related to the chemical engineering field and process design.
Currently, the software can assist users with the analysis of heat exchange networks (HENs) and water recycle networks (WReNs). Other relevant processes, such as distillation and reactors, will be implemented in the future.\
ALChemE can be operated via Terminal / IDE / Jupyter Notebooks or via a GUI. This README will concern itself with the former, while [User Manual.pdf](User%20Manual.pdf) will concern itself with the latter.

## 1) Heat Exchanger Networks (HENs)
HEN analysis can assist users in maximizing energy recovery and minimizing costs.
Currently, users can input streams and utilities (with units of measurement), plot temperature interval diagrams and composite curves, and manually add heat exchangers.
The program automatically handles unit conversions, calculates properties such as MER targets or the pinch point location, and can automatically place exchangers to minimize their number.
This function also returns the cost of the HENs generated.\
To begin, import the HEN class from the [hen_design.py](hen_design.py) module. We will also import the unyt module to easily change the default units if needed.
```python
from hen_design import HEN
import unyt as u
```
Then, create a HEN object from that class. This object contains all other relevant methods for HEN analysis, including solving the HEN and making plots. Note that you can select the minimum ΔT value and the default units when creating the HEN object.
```python
myhen = HEN() # Uses the default ΔT of 10 °C and the default units of flow [=] kg/s, temperature [=] °C, and c_p [=] J / (°C * kg)
myhen = HEN(20) # Uses a ΔT of 20 °C and the default units
myhen = HEN(cp_unit = u.kJ / (u.delta_degC * u.kg)) # Uses the default ΔT of 10 °C, but changes the c_p units
myhen = HEN(30, temp_unit = u.delta_degF, cp_unit = u.J / (u.delta_degF * u.kg)) # Uses a ΔT of 30 °F and changes the c_p units
```
Any streams or utilities added will inherit the default units of the HEN object if created without units, or will have their units converted automatically to the default units of the HEN object if created with units. For all of the following code, assume the HEN object was created with the default units listed above.\
To add a stream, use the `HEN.add_stream()` method:
```python
myhen.add_stream(360, 180, 3) # Creates a hot stream with T_in = 360 °C, T_out = 180 °C, and F*c_p = 3 J / (°C * s)
myhen.add_stream(45, 60, 2) # Creates a cold stream with T_in = 45 °C, T_out = 60 °C, and F*c_p = 2 J / (°C * s)

# One may separate the flow rate and the c_p components, if desired. The following line is typically equivalent to myhen.add_stream(95, 75, 50):
myhen.add_stream(95, 75, 5, 10) # Creates a hot stream with T_in = 95 °C, T_out = 75 °C, c_p = 5 J / (°C * kg), and flow = 10 kg / s

# One may create a stream with units, which will be automatically converted to the HEN object's units:
myhen.add_stream(100, 180, 2, cp_unit = u.kJ / (u.delta_degC * u.kg)) # Equivalent to myhen.add_stream(100, 180, 2000) if the HEN's cp_unit is J / (°C * kg)
myhen.add_stream(338, 401, 3.5, temp_unit = u.delta_degF) # Equivalent to myhen.add_stream(170, 205, 3.5) if the HEN's temp_unit is °C
```
To add a utility, use the `HEN.add_utility()` method. Currently, utilities are assumed to not change temperature, and thus only one temperature value can be passed. Utilities also take a cost parameter.
```python
myhen.add_utility('hot', 300, 0.70) # Creates a hot utility with T = 300 °C and cost = $0.70 / W
myhen.add_utility('cold', 20, 0.05) # Creates a cold utility with T = 20 °C and cost = $0.05 / W

# As with streams, one may use different units, which will be automatically converted:
myhen.add_utility('hot', 500, 1, temp_unit = u.delta_degF) # Equivalent to myhen.add_utility('hot', 260, 1) if the HEN's temp_unit is °C
```
After adding all relevant streams and utilities, the user should run the `HEN.get_parameters()` method. This method calculates the pinch point location and the minimum utilities required, and generates essential information other methods require. **Other methods will fail if `HEN.get_parameters()` is not run, or if a stream/utility is added and `HEN.get_parameters()` is not run again.**\
Next, the user may use the plotting tools or the solver. The plotting tools can generate temperature interval diagrams (TIDs) and composite curves (CCs). TIDs show some extra information about the system, and the user may hide these during the function call:
```python
# TIDs
myhen.make_tid() # TID with temperature and stream property information
myhen.make_tid(show_temperatures = False, show_properties = False) # TID without any information

# CCs
myhen.make_cc() # No information customization is available for the CCs
```
The solver method, `HEN.solve_HEN()` will automatically generate stream matches over a subnetwork (that is, above or below pinch). In its most basic form, the software calculates the maximum heat that can be exchanged between any two streams automatically and returns the location of, the heat exchanged within, and the cost of each exchanger. The user can manually set upper or lower limits on each match (for example, to prevent very small heat exchangers from being used, or to forbid a match by settting its upper limit = 0). The user can also require certain matches without demanding a lower heat limit. No HEN may exist which simultaneously obeys the MER and the restrictions made by the user, which will cause the program to return an error. To add any constraint, either generate your own array-like object and pass it to `HEN.solve HEN()` or edit the pre-built arrays as shown below:
```python
# Constraint arrays - for ease-of-use, edit these and pass them to myhen.solve_HEN()
myhen.upper_limit
myhen.lower_limit
myhen.required

# Solving the network without any constraints
myhen.solve_HEN('above')
myhen.solve_HEN('below')

# Solving with some constraints
myhen.solve_HEN('above', upper = myhen.upper_limit)
myhen.solve_HEN('above', lower = myhen.lower_limit)
myhen.solve_HEN('above', required = myhen.required)
```
Results are automatically added to the `myhen.results_above` or `myhen.results_below` objects. The results will be in a 2-tiered Pandas DataFrame, where the first tier represents whether the results indicate heat values or dollar cost values, and the second tier represent the actual match results. Costs are generated assuming U = 100 J / (°C * m^2 * s) and fixed head exchangers, but this behavior can be changed by changing the parameters passed when calling `myhen.solve_HEN()`.
```python
# After running myhen.solve_HEN('above') with or without any constraints
myhen.results_above.loc['Q'] # Returns the heat exchanged in each match
myhen.results_above.loc['costs'] # Returns the cost of each match

# After running myhen.solve_HEN('below') with or without any constraints
myhen.results_below.loc['Q'] # Returns the heat exchanged in each match
myhen.results_below.loc['costs'] # Returns the cost of each match
```
Currently, this solver focuses on minimizing the number of exchangers while respecting the MER. Thus, there may be multiple valid solutions with the same number of exchangers. To attempt to increase the number of solutions returned, use the _depth_ parameter. A depth of 0 returns only one solution, while higher depths may return more. While there is no guarantee the software will find all possible solutions, this can provide alternative HENs, some of which may be better than the first HEN found.
```python
# Solving the network without any constraints
myhen.solve_HEN('above', depth = 2)
myhen.solve_HEN('below', depth = 3)

# Depth is compatible with constraints
myhen.solve_HEN('above', lower = myhen.lower_limit, depth = 3)
myhen.solve_HEN('below', required = myhen.required, depth = 1)

# A simple way to view all solutions at once.
# Replace results_above with results_below if viewing solutions below the pinch point.
for idx, sol in enumerate(myhen.results_above):
    print(f'Solution {idx+1}')
    print(sol.loc['Q'])
    print(sol.loc['cost'])
    print()
```

## 2) Water Recycle Networks (WReNs)
To be added.