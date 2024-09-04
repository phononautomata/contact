import json
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import norm

cwd_path = os.getcwd()

default_country = 'United_States'
default_year = 2020

def age_group_contact_number_distribution(contact):
    return np.sum(contact, axis=1) #* pop_a / np.sum(pop_a)

def check_symmetry_condition(contact, population):
    contact = np.array(contact)
    population = np.array(population)

    if contact.shape[0] != contact.shape[1]:
        raise ValueError("Contact matrix must be square.")
    if len(population) != contact.shape[0]:
        raise ValueError("Population array size must match the dimensions of the contact matrix.")

    left_side = contact * population[:, np.newaxis]  # contact_{ij} * population_i
    right_side = contact.T * population  # contact_{ji} * population_j

    return np.allclose(left_side, right_side)

def coarse_grain_population(population):
    if len(population) != 85:
        raise ValueError("Population vector must have 85 elements.")

    underage_range = range(0, 19)
    adults_range = range(19, 65)
    elders_range = range(65, 85)

    underage_population = np.sum(population[underage_range])
    adults_population = np.sum(population[adults_range])  
    elders_population = np.sum(population[elders_range])  

    cg_population = np.array([underage_population, adults_population, elders_population])

    return cg_population

def coarse_grain_contact_matrix(contact, population):
    contact = np.array(contact)
    population = np.array(population)

    if contact.shape != (85, 85):
        raise ValueError("Contact matrix must be of size 85x85.")
    if len(population) != 85:
        raise ValueError("Population vector must have 85 elements.")

    underage_range = range(0, 19)  
    adults_range = range(19, 65)
    elders_range = range(65, 85)

    cg_population = [
        np.sum(population[underage_range]),
        np.sum(population[adults_range]),
        np.sum(population[elders_range]) 
    ]

    cg_contact = np.zeros((3, 3))

    group_ranges = [underage_range, adults_range, elders_range]

    for i_prime in range(3):
        for j_prime in range(3):
            i_range = group_ranges[i_prime]
            j_range = group_ranges[j_prime]
            
            weighted_sum = np.sum(
                contact[np.ix_(i_range, j_range)] * population[i_range, np.newaxis]
            )
            
            cg_contact[i_prime, j_prime] = weighted_sum / cg_population[i_prime]

    return cg_contact

def collect_age_structure_data(path=cwd_path, lower_path='data'):
    state_list = get_state_list()

    population_dict = {}
    contact_dict = {}
    degree_distribution_dict = {}
    degree_distribution_age_dict = {}

    for state in state_list:
        # Reference data
        pop_a = import_age_distribution(state=state)
        contact = import_contact_matrix(state=state)

        # Updated data
        new_pop_a = import_age_distribution(state=state, reference=False, year=2019)
        new_contact = update_contact_matrix(contact, old_pop_a=pop_a, new_pop_a=new_pop_a)

        # Age-group degree distribution
        degree_pdf = average_degree_distribution(new_contact, new_pop_a)

        # Intralayer degree distribution

        #population_dict[state] = new_pop_a
        #contact_dict[state] = new_contact
        #degree_distribution_dict[state] = degree_distribution

        if np.sum(new_pop_a) != 1.0:
            new_pop_a /= np.sum(new_pop_a)
        if np.sum(degree_pdf) != 1.0:
            degree_pdf / np.sum(degree_pdf)

        population_dict[state] = new_pop_a.tolist() if isinstance(new_pop_a, np.ndarray) else new_pop_a
        contact_dict[state] = new_contact.tolist() if isinstance(new_contact, np.ndarray) else new_contact
        degree_distribution_dict[state] = degree_pdf.tolist() if isinstance(degree_pdf, np.ndarray) else degree_pdf

    file_name = 'state_population_age.json'
    full_name = os.path.join(path, lower_path, file_name)
    with open(full_name, 'w') as file:
        json.dump(population_dict, file)

    file_name = 'state_contact_matrix.json'
    full_name = os.path.join(path, lower_path, file_name)
    with open(full_name, 'w') as file:
        json.dump(contact_dict, file)

    file_name = 'state_degree.json'
    full_name = os.path.join(path, lower_path, file_name)
    with open(full_name, 'w') as file:
        json.dump(degree_distribution_dict, file)

def count_fraction_underage(array_population):
    return np.sum(array_population[0:18])

def average_degree_distribution(contact, pop_a, norm_flag=False):
    degree_a = np.sum(contact, axis=1)
    pdf = np.zeros(len(degree_a))

    for a in range(len(degree_a)):
        k = round(degree_a[a])
        pdf[k] += pop_a[a]

    if norm_flag:
        pdf /= np.sum(pdf)

    return pdf

def get_state_list():
    return ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'Delaware', 'District_of_Columbia', 
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
              'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
              'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New_Hampshire',  
              'New_Jersey', 'New_Mexico', 'New_York', 'North_Carolina', 
              'North_Dakota', 'Ohio', 'Oklahoma', 'Oregon', 
              'Pennsylvania', 'Rhode_Island', 'South_Carolina', 'South_Dakota', 
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 
              'Washington', 'West_Virginia', 'Wisconsin', 'Wyoming', 'National']

def import_age_distribution(
        id_state, 
        path=cwd_path, 
        lower_path='data/raw', 
        country=default_country, 
        reference=True, 
        year=default_year,
        norm_flag=False,
        ):
    if reference:
        if id_state == 'National':
            file_name = country + '_country_level_' + 'age_distribution_85' 
        else:
            file_name = country + '_subnational_' + id_state + '_age_distribution_85'
        extension = '.csv'
        full_name = os.path.join(path, lower_path, file_name + extension)

        age_df = pd.read_csv(full_name, header=None)

        pop_a = age_df.values.T[1]
    else:
        if id_state == 'National':
            file_name = str(year) + '_' + country + '_country_level_' + 'age_distribution_85'
            extension = '.xlsx'
            full_name = os.path.join(path, lower_path, file_name + extension)

            full_age_df = pd.read_excel(full_name)

            age_df = full_age_df['Unnamed: 12'][4:89].values
            merge_older = np.sum(full_age_df['Unnamed: 12'][88:105].values)
            age_df[-1] = merge_older

            pop_a = np.zeros(len(age_df), dtype=float)
            for a in range(len(age_df)):
                pop_a[a] = age_df[a]

        else:
            file_name = str(year) + '_' + country + '_subnational_' + id_state + '_age_distribution_85'
            extension = '.xlsx'
            full_name = os.path.join(path, lower_path, file_name + extension)

            full_age_df = pd.read_excel(full_name)

            age_df = full_age_df['Unnamed: 34'][5:90].values
            merge_older = full_age_df['Unnamed: 34'][89] + full_age_df['Unnamed: 34'][90]
            age_df[-1] = merge_older

            pop_a = np.zeros(len(age_df), dtype=float)
            for a in range(len(age_df)):
                pop_a[a] = age_df[a]

    if norm_flag:
        pop_a /= np.sum(pop_a)

    return pop_a

def import_contact_matrix(
        id_state, 
        path=cwd_path, 
        lower_path='data/raw', 
        country=default_country
        ):
    if id_state == 'National':
        file_name = country + '_country_level_' + 'M_overall_contact_matrix_85'
    else:
        file_name = country + '_subnational_' + id_state + '_M_overall_contact_matrix_85'
    extension = '.csv'
    full_name = os.path.join(path, lower_path, file_name + extension)

    contact_df = pd.read_csv(full_name, header=None)

    return contact_df.values

def update_contact_matrix(contact, old_pop_a, new_pop_a):
    N_old = np.sum(old_pop_a)
    N_new = np.sum(new_pop_a)
    A = len(old_pop_a)
    new_contact = np.zeros((A, A), dtype=float)

    for i in range(A):
        for j in range(A):
            old_fraction = N_old / old_pop_a[j]
            new_fraction = new_pop_a[j] / N_new
            factor = old_fraction * new_fraction
            new_contact[i][j] = contact[i][j] * factor

    return new_contact
