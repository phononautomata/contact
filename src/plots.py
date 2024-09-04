import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_contact(contact, id_state, flag_log=False):
    contact = np.array(contact)

    if contact.shape[0] != contact.shape[1]:
        raise ValueError("Contact matrix must be square.")
    
    fig, ax = plt.subplots(figsize=(12, 8))

    if flag_log:
        cax = ax.imshow(contact, cmap='viridis', norm=LogNorm())
    else:
        cax = ax.imshow(contact, cmap='viridis')

    cb = fig.colorbar(cax, ax=ax)
    cb.ax.tick_params(labelsize=14)

    if flag_log:
        cb.set_label('Average per capita contacts (log-scale)', fontsize=18)
    else:
        cb.set_label('Average per capita contacts', fontsize=18)

    ax.set_title('Contact matrix for {0}'.format(id_state), fontsize=20)
    ax.set_xlabel(r'Age group', fontsize=18)
    ax.set_ylabel(r'Age group', fontsize=18)
    ax.xaxis.set_ticks_position('bottom')
        
    tick_indices = np.arange(0, len(contact), 5)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)

    ax.set_xticklabels(tick_indices)
    ax.set_yticklabels(tick_indices)
      
    ax.grid(False)

    plt.show()

def plot_total_group_average_contacts(contact):
    contact = np.array(contact)

    row_sums = np.sum(contact, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    age_groups = range(len(contact))
    ax.bar(age_groups, row_sums, color='skyblue', edgecolor='black')

    ax.set_title('Sum of average contacts', fontsize=16)
    ax.set_xlabel('Age Group', fontsize=14)
    ax.set_ylabel('Total', fontsize=14)

    tick_indices = np.arange(0, len(contact), 5)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_indices, rotation=45)

    fig.tight_layout()

    plt.show()