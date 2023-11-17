# Name: Lankun Chen, Miji Trenkel 
# Student number: 13591509, 13686038
# Date : 04-Nov-2023
#
# This program can simulate a one-dimensional cellular automata, and simulate the 
# average cycle length of the cellular automata under rule 0-255 when k=2, r=1.

import numpy as np
from pyics import Model
import matplotlib.pyplot as plt
import pandas as pd

def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""

    result_list = []

    while n > 0:
        r = n % k
        n = n // k                       # Integer division
        result_list.insert(0, r) # Get remainder and store it in the beginning of the list

    return result_list

class CASim(Model):
    def __init__(self):
        Model.__init__(self)
        self.t = 0
        self.rule_set = []
        self.config = None
        self.random_seed = True         # A random seed is used by default.

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('width', 50)
        self.make_param('height', 50)
        self.make_param('rule', 30, setter=self.setter_rule)

    def setter_rule(self, val):
        """Setter for the rule parameter, clipping its value between 0 and the
        maximum possible rule number."""
        rule_set_size = self.k ** (2 * self.r + 1)
        max_rule_number = self.k ** rule_set_size
        return max(0, min(val, max_rule_number - 1))

    def build_rule_set(self):
        """Sets the rule set for the current rule.
        A rule set is a list with the new state for every old configuration.

        For example, for rule=34, k=3, r=1 this function should set rule_set to
        [0, ..., 0, 1, 0, 2, 1] (length 27). This means that for example
        [2, 2, 2] -> 0 and [0, 0, 1] -> 2."""

        # Define the size of input alphabet
        size_input = self.k ** (2 * self.r + 1)

        # Convert rule n to base-k
        bask_k_result_list = decimal_to_base_k(self.rule, self.k)

        # Extend the transformation result to the size of input
        while len(bask_k_result_list) < size_input:
            bask_k_result_list.insert(0,0)
        
        self.rule_set = bask_k_result_list

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
        base_10_number = 0

        # Convert the inp into number 10-based
        for i in range(len(inp)):
            base_10_number += inp[i] * self.k ** (len(inp) -i -1)
        
        # Return corresponding state
        return self.rule_set[::-1][int(base_10_number)]

    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        
        # Use randwom seed to set the initial state by default
        if self.random_seed == True:
            arr = np.random.randint(0, self.k, size=self.width, dtype=int)
            return arr
        
        # If self.rand_seed is False, set the initial state by single seed
        else:
            arr = np.zeros(self.width, dtype=int)
            middle_index = self.width //2
            arr[middle_index] = 1
            return arr

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        import matplotlib
        import matplotlib.pyplot as plt

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=matplotlib.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
        self.t += 1
        if self.t >= self.height:
            return True

        for patch in range(self.width):
            # We want the items r to the left and to the right of this patch,
            # while wrapping around (e.g. index -1 is the last item on the row).
            # Since slices do not support this, we create an array with the
            # indices we want and use that to index our grid.
            indices = [i % self.width
                    for i in range(patch - self.r, patch + self.r + 1)]
            values = self.config[self.t - 1, indices]
            self.config[self.t, patch] = self.check_rule(values)
        
    def find_cycle_length(self):
        """
        Find the cycle length.
        """
        # Create an empty dictionary to store the cell state at time t.
        state_dict= {}

        while True:
            current_state = tuple(self.config[self.t])
            if current_state in state_dict:
                # Repeats occur and periodicity is found.
                return self.t - state_dict[current_state]
            else:
                state_dict[current_state] = self.t

                # If it is not cyclical, return None.
            if self.step():
                return None


# Load the CSV file containing the rules and their corresponding classes
rules_classes_df = pd.read_csv('ca/rule_class_wolfram.csv')
rules_classes_df.columns = ['rule', 'class']

def find_avg_cycle_length(model, model_size, total_experiments, max_step=1e4, rules_classes_df=rules_classes_df):
    """
    This function can draw the scatter plot of cycle length corresponding to rule 0-255 and 
    error bar of average cycle length based on the input 1-D cellular automata and its model size.
    It uses the rules_classes_df dataframe to plot the cycle lengths for rules grouped by their class.
    """
    # Group the rules by class
    grouped_rules = rules_classes_df.groupby('class')['rule'].apply(list).to_dict()

    # Create an empty dictionary to store cycle lengths
    cycle_length_dict_by_class = {class_: {} for class_ in grouped_rules.keys()}

    for class_, rules in grouped_rules.items():
        for rule in rules:
            cycle_length_dict_by_class[class_][rule] = []

            # Repeat the simulation and save the corresponding cycle length
            for _ in range(total_experiments):
                model.rule = rule
                model.width = model_size
                model.height = int(max_step)
                model.reset()

                # Get the cycle_length
                cycle_length = model.find_cycle_length()

                # If have cycle length, append it in corresponding list in a dictionary
                if cycle_length:
                    cycle_length_dict_by_class[class_][rule].append(cycle_length)

    # Set the size of the figure
    plt.figure(figsize=(20, 8))

    # Define colors for each class
    colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple'}
    legends_added = {1: False, 2: False, 3: False, 4: False}

    for class_, cycle_length_dict in cycle_length_dict_by_class.items():
        rules = list(cycle_length_dict.keys())

        # Calculate the mean and standard deviation of the cycle length, 
        # replace None with np.nan for proper handling in plots
        mean_cycle_lengths = [np.mean(cycle_length_dict[rule]) if cycle_length_dict[rule] else np.nan for rule in rules]
        sd_cycle_lengths = [np.std(cycle_length_dict[rule]) if cycle_length_dict[rule] else np.nan for rule in rules]

        # Filter out any `None` or `NaN` values before plotting
        valid_indices = ~np.isnan(sd_cycle_lengths)
        valid_rules = np.array(rules)[valid_indices]
        valid_means = np.array(mean_cycle_lengths)[valid_indices]
        valid_sds = np.array(sd_cycle_lengths)[valid_indices]

        # Only plot error bars for valid entries
        plt.errorbar(valid_rules, valid_means, yerr=valid_sds, fmt='o', color=colors[class_],
                     ecolor='black', capsize=1, elinewidth=0.5, markersize=2, label=f'Average cycle length (Class {class_})')

    # Draw pictures
    plt.xlabel('Rule number')
    plt.ylabel('Cycle length')
    plt.title(f'Average cycle lengths with error bars and scatter points of cycle length by class with model size {model_size} ({total_experiments} simulations)')
    plt.xticks(range(0, 256, 5), fontsize=8, rotation=90)
    plt.xlim(-1, 256)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    """
    In this program we set the initial state to a random seed. 
    If you want to use a single seed (initial state is only 1 in the middle of cell), please set:
    "sim.random_seed = False"
    If you would like to plot the graph of average_cycle_length graph, please comment the GUI codes.
    """
    sim = CASim()
    # sim.random_seed = False
    # from pyics import GUI
    # cx = GUI(sim)
    # cx.start()

    find_avg_cycle_length(sim, model_size=15, total_experiments=100)
