import pandas as pd
import scipy.stats as st

fulldata = pd.read_csv(fr"datasets\vehicle_claim_fraud.csv", index_col='PolicyNumber')

standard_vars = ['Age', 'Deductible']

binary_vars = ['Sex', 'PoliceReportFiled', 'WitnessPresent',
               'AgentType', 'Fault', 'AccidentArea']

ordinal_vars = ['Month', 'MonthClaimed', 'DayOfWeek', 'DayOfWeekClaimed',
                 'VehiclePrice', 'Days_Policy_Accident','Days_Policy_Claim',
                 'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder',
                 'NumberOfSuppliments', 'NumberOfCars', 'Year']

categorical_vars = ['Make', 'MaritalStatus', 'PolicyType',
                    'VehicleCategory', 'BasePolicy']

month_keys = list(fulldata['Month'].unique())
month_vals = [11, 0, 9, 5, 1, 10, 3, 2, 7, 6, 4, 8]
month_map = dict(zip(month_keys, month_vals))

day_keys = list(fulldata['DayOfWeek'].unique())
day_vals = [2, 4, 5, 0, 1, 6, 3]
day_map = dict(zip(day_keys, day_vals))

price_keys = list(fulldata['VehiclePrice'].unique())
price_vals = [5, 1, 2, 0, 3, 4]
price_map = dict(zip(price_keys, price_vals))

dayp_keys = list(fulldata['Days_Policy_Accident'].unique())
dayp_vals = [4, 3, 0, 1, 2]
dayp_map = dict(zip(dayp_keys, dayp_vals))

pnc_keys = list(fulldata['PastNumberOfClaims'].unique())
pnc_vals = [0, 1, 2, 3]
pnc_map = dict(zip(pnc_keys, pnc_vals))

agev_keys = list(fulldata['AgeOfVehicle'].unique())
agev_vals = [2, 5, 6, 7, 4, 0, 3, 1]
agev_map = dict(zip(agev_keys, agev_vals))

ageph_keys = list(fulldata['AgeOfPolicyHolder'].unique())
ageph_vals = [3,4,6,7,2,5,0,8,1]
ageph_map = dict(zip(ageph_keys, ageph_vals))

supn_keys = list(fulldata['NumberOfSuppliments'].unique())
supn_vals = [0, 3, 2, 1]
supn_map = dict(zip(supn_keys, supn_vals))

carn_keys = list(fulldata['NumberOfCars'].unique())
carn_vals = [2, 0, 1, 3, 4]
carn_map = dict(zip(carn_keys, carn_vals))

year_keys = list(fulldata['Year'].unique())
year_vals = [0, 1, 2]
year_map = dict(zip(year_keys, year_vals))

acc_keys = list(fulldata['AddressChange_Claim'].unique())
acc_vals = [2, 0, 4, 3, 1]
acc_map = dict(zip(acc_keys, acc_vals))

sex_map = {
    "Female" : 0,
    "Male" : 1
}

yn_map = {
    "No" : 0,
    "Yes" : 1
}

at_map = {
    "Internal" : 0,
    "External" : 1
}

flt_map = {
    "Policy Holder" : 1,
    "Third Party" : 0
}

aa_map = {
    "Urban" : 0,
    "Rural" : 1
}

pt_keys = list(fulldata['PolicyType'].unique())
pt_vals = list(fulldata['PolicyType'].unique())

for i in range(len(pt_vals)):
    pt_vals[i] = pt_vals[i].replace(' - ', '_').replace(' ', '')
    
pt_map = dict(zip(pt_keys, pt_vals))

bp_keys = list(fulldata['BasePolicy'].unique())
bp_vals = list(fulldata['BasePolicy'].unique())

for i in range(len(bp_vals)):
    bp_vals[i] = bp_vals[i].replace(' ', '')
    
bp_map = dict(zip(bp_keys, bp_vals))

fulldata[categorical_vars] = fulldata[categorical_vars].astype('category')

fulldata['Month'] = fulldata['Month'].replace(month_map)
fulldata['MonthClaimed'] = fulldata['MonthClaimed'].replace(month_map)
fulldata['DayOfWeek'] = fulldata['DayOfWeek'].replace(day_map)
fulldata['DayOfWeekClaimed'] = fulldata['DayOfWeekClaimed'].replace(day_map)
fulldata['VehiclePrice'] = fulldata['VehiclePrice'].replace(price_map)
fulldata['Days_Policy_Accident'] = fulldata['Days_Policy_Accident'].replace(dayp_map)
fulldata['Days_Policy_Claim'] = fulldata['Days_Policy_Claim'].replace(dayp_map)
fulldata['PastNumberOfClaims'] = fulldata['PastNumberOfClaims'].replace(pnc_map)
fulldata['AgeOfVehicle'] = fulldata['AgeOfVehicle'].replace(agev_map)
fulldata['AgeOfPolicyHolder'] = fulldata['AgeOfPolicyHolder'].replace(ageph_map)
fulldata['NumberOfSuppliments'] = fulldata['NumberOfSuppliments'].replace(supn_map)
fulldata['NumberOfCars'] = fulldata['NumberOfCars'].replace(carn_map)
fulldata['Year'] = fulldata['Year'].replace(year_map)
fulldata['AddressChange_Claim'] = fulldata['AddressChange_Claim'].replace(acc_map)
fulldata['Sex'] = fulldata['Sex'].replace(sex_map)
fulldata['PoliceReportFiled'] = fulldata['PoliceReportFiled'].replace(yn_map)
fulldata['WitnessPresent'] = fulldata['WitnessPresent'].replace(yn_map)
fulldata['AgentType'] = fulldata['AgentType'].replace(at_map)
fulldata['Fault'] = fulldata['Fault'].replace(flt_map)
fulldata['AccidentArea'] = fulldata['AccidentArea'].replace(aa_map)
fulldata['PolicyType'] = fulldata['PolicyType'].replace(pt_map)
fulldata['BasePolicy'] = fulldata['BasePolicy'].replace(bp_map)

fulldata['Age'] = st.zscore(fulldata['Age'])
fulldata['Deductible'] =st.zscore(fulldata['Deductible'])

prefix_vals = []

for i in categorical_vars:
    prefix_vals.append(i + "Is")

prefixes = dict(zip(categorical_vars, prefix_vals))

fulldata = pd.get_dummies(fulldata, prefixes, '', columns = categorical_vars)

label = fulldata.pop('FraudFound_P')

fulldata['Fraud'] = label

fulldata = fulldata.astype(float)

fulldata.to_csv(fr'datasets\fraud_processed.csv')