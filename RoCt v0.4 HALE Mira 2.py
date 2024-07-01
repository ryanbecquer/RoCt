from RocketFunc import *

import csv

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity

# extract all species in the NASA database
full_species = {S.name: S for S in ct.Species.list_from_file('nasa_gas.yaml')}


with open('RoCt Input Hale.yaml', 'r') as f:
    input_file = yaml.safe_load(f)


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
o_f_ratio = input_file['OF_ratio']
temperature_fuel = Q_(293.15, 'K')
temperature_oxidizer = Q_(90.170, 'K')
chamber_pressure = Q_(input_file['Pressure'], 'psi')


fuel = ct.Solution('propellants.yaml', 'liquid_jetA')
fuel.TP = to_si(temperature_fuel), to_si(chamber_pressure)

oxidizer = ct.Solution('propellants.yaml', 'liquid_oxygen')
oxidizer.TP = to_si(temperature_oxidizer), to_si(chamber_pressure)

molar_ratio = o_f_ratio / (oxidizer.mean_molecular_weight / fuel.mean_molecular_weight)
moles_ox = molar_ratio / (1 + molar_ratio)
moles_f = 1 - moles_ox

species_list = [
    'C2H2,acetylene', 'O2', 'H2O', 'CO', 'CO2', 'H2O2', 'OH', 'H',
    'CH4', 'C2H4', 'H2',
    ]

# Get all of the Species objects defined
species = {S.name: S for S in ct.Species.list_from_file('nasa_gas.yaml')}

# Create an IdealGas object with all species involved
gas_chamber = ct.Solution(
    thermo='ideal-gas',
    species=[species[S] for S in species_list]
    )

# create a mixture of the liquid phases with the gas-phase model,
# with the number of moles for fuel and oxidizer based on
# the O/F ratio
mix = ct.Mixture([(gas_chamber, 0), (fuel, moles_f), (oxidizer, moles_ox)])

# Solve for the equilibrium state, at constant enthalpy and pressure
mix.equilibrate('HP')

gas_chamber()

derivs = get_thermo_derivatives(gas_chamber)

dlogV_dlogT_P, dlogV_dlogP_T, cp, gamma_s = get_thermo_properties(gas_chamber, derivs[0], derivs[1], derivs[2])

print(f'Cp = {cp: .2f} J/(K kg)')

print(f'(d log V/d log P)_T = {dlogV_dlogP_T: .4f}')
print(f'(d log V/d log T)_P = {dlogV_dlogT_P: .4f}')

print(f'gamma_s = {gamma_s: .4f}')

speed_sound = np.sqrt(ct.gas_constant * gas_chamber.T * gamma_s / gas_chamber.mean_molecular_weight)
print(f'Speed of sound = {speed_sound: .1f} m/s')
print()

entropy_chamber = gas_chamber.s
enthalpy_chamber = gas_chamber.enthalpy_mass
mole_fractions_chamber = gas_chamber.X
gamma_chamber = gamma_s

c_star = calculate_c_star(gamma_chamber, gas_chamber.T, gas_chamber.mean_molecular_weight)
print(f'c-star: {c_star: .1f} m/s')
print()


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Create an IdealGas object with all species involved
gas_throat = ct.Solution(
    thermo='ideal-gas', 
    species=[species[S] for S in species_list]
    )

pressure_throat = chamber_pressure / np.power((gamma_chamber + 1) / 2., gamma_chamber / (gamma_chamber - 1))

# based on CEA defaults
max_iter_throat = 5
tolerance_throat = 0.4e-4

mach = 1.0
num_iter = 0
residual = 1
while residual > tolerance_throat:
    num_iter += 1
    if num_iter == max_iter_throat:
        break
        #print(f'Error: more than {max_iter_throat} iterations required for throat calculation')
    pressure_throat = pressure_throat * (1 + gamma_s * mach**2) / (1 + gamma_s)
    
    gas_throat.SPX = entropy_chamber, to_si(pressure_throat), mole_fractions_chamber
    gas_throat.equilibrate('SP')

    derivs = get_thermo_derivatives(gas_throat)
    dlogV_dlogT_P, dlogV_dlogP_T, cp, gamma_s = get_thermo_properties(gas_throat, derivs[0], derivs[1], derivs[2])
    
    velocity = np.sqrt(2 * (enthalpy_chamber - gas_throat.enthalpy_mass))
    speed_sound = np.sqrt(ct.gas_constant * gas_throat.T * gamma_s / gas_throat.mean_molecular_weight)
    mach = velocity / speed_sound

    residual = np.abs(1.0 - 1/mach**2)

temperature_throat = gas_throat.T
pressure_throat = Q_(gas_throat.P, 'Pa')
gamma_s_throat = gamma_s
print()
print(f'Temperature at throat: {temperature_throat:.2f} K, Pressure at throat: {pressure_throat.to("bar"):.2f}')
print()
gas_throat()


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
area_ratio = input_file['Area_Ratio']
# this is constant
A_mdot_thr = gas_throat.T / (gas_throat.P * velocity * gas_throat.mean_molecular_weight)

#gas_exit = ct.Solution('nasa_h2o2.yaml', 'gas')

# Create an IdealGas object with all species involved
# Create an IdealGas object with all species involved
gas_exit = ct.Solution(
    thermo='ideal-gas', 
    species=[species[S] for S in species_list]
    )

gas_exit.SPX = gas_throat.s, gas_throat.P, gas_throat.X

# initial estimate for pressure ratio
pinf_pe = np.exp(gamma_s_throat + 1.4 * np.log(area_ratio))
p_exit = to_si(chamber_pressure) / pinf_pe

gas_exit.SP = entropy_chamber, p_exit
gas_exit.equilibrate('SP')

Ae_At = gas_exit.T / (gas_exit.P * velocity * gas_exit.mean_molecular_weight) / A_mdot_thr

num_iter = 0
print('Iter  T_exit   Ae/At    P_exit     P_inf/P')
print(f'{num_iter}  {gas_exit.T:.3f} K   {Ae_At: .2f}  {gas_exit.P/1e5:.3f} bar  {pinf_pe:.3f}')
max_iter_exit = 10
tolerance_exit = 4e-5

residual = 1
while np.abs(residual) > tolerance_exit:
    num_iter += 1
    
    if num_iter == max_iter_throat:
        break
        print(f'Error: more than {max_iter_exit} iterations required for exit calculation')

    derivs = get_thermo_derivatives(gas_exit)
    dlogV_dlogT_P, dlogV_dlogP_T, cp, gamma_s = get_thermo_properties(gas_exit, derivs[0], derivs[1], derivs[2])
    velocity = np.sqrt(2 * (enthalpy_chamber - gas_exit.enthalpy_mass))
    speed_sound = np.sqrt(ct.gas_constant * gas_exit.T * gamma_s / gas_exit.mean_molecular_weight)

    Ae_At = gas_exit.T / (gas_exit.P * velocity * gas_exit.mean_molecular_weight) / A_mdot_thr

    dlogp_dlogA = gamma_s * velocity**2 / (velocity**2 - speed_sound**2)
    residual = dlogp_dlogA * (np.log(area_ratio) - np.log(Ae_At))
    log_pinf_pe = np.log(pinf_pe) + residual

    pinf_pe = np.exp(log_pinf_pe)
    p_exit = to_si(chamber_pressure) / pinf_pe

    gas_exit.SP = entropy_chamber, p_exit
    gas_exit.equilibrate('SP')
    
    print(f'{num_iter}  {gas_exit.T:.3f} K  {Ae_At: .2f}  {gas_exit.P/1e5:.3f} bar  {pinf_pe:.3f}')
print()
print(f'Exit pressure: {Q_(gas_exit.P, "Pa").to("bar"): .5f~P}')
print(f'Exit temperature: {gas_exit.T: .2f} K')
print()

derivs = get_thermo_derivatives(gas_exit)
dlogV_dlogT_P, dlogV_dlogP_T, cp, gamma_s = get_thermo_properties(gas_exit, derivs[0], derivs[1], derivs[2])
velocity = np.sqrt(2 * (enthalpy_chamber - gas_exit.enthalpy_mass))

thrust_coeff = velocity / c_star
print(f'Thrust coefficient: {thrust_coeff: .4f}')

g0 = 9.80665
Isp = velocity
Ivac = Isp + gas_exit.T * ct.gas_constant / (velocity * gas_exit.mean_molecular_weight)
#print(f'I_vac = {Ivac: .1f} m/s')
#print(f'I_sp = {Isp: .1f} m/s')

print()
print('Specific impulse:')
print(f'Vacuum Isp = {Ivac / g0: .1f} s')
print(f'Sea Level Isp = {Isp / g0: .1f} s')
print()
gas_exit()

#Save data to csv
with open('data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([area_ratio, c_star, thrust_coeff, Ivac / g0, Isp / g0])