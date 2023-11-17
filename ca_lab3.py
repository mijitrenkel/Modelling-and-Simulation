import numpy as np
import itertools
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from pyics import Model
import math 


def decimal_to_base_k(n, k):
    """Converts a given decimal (i.e. base-10 integer) to a list containing the
    base-k equivalant.

    For example, for n=34 and k=3 this function should return [1, 0, 2, 1]."""
    
    length = k**(2*1+1)
    rule_set = [0]*length
    rule = n
    for i in range(length-1,-1, -1):
        if n >= k**i:
            amount_fitted = rule // k**i
            rule -= amount_fitted * k**i
            rule_set[length-i-1] = amount_fitted
    
    trimmed_rule_set = []
    found_non_zero = False
    
    for num in rule_set:
        if num!= 0:
            found_non_zero = True
        if found_non_zero == True:
            trimmed_rule_set.append(num)
    
    return trimmed_rule_set
    

class CASim(Model):
    def __init__(self, width=50, height=50,rule_value=4, random_seed=0):
        Model.__init__(self)
        
        self.cycle_length = 0
        self.state_history= []
        self.t = 0
        self.rule_set = []
        self.config = None
        
        random.seed(random_seed)

        self.make_param('r', 1)
        self.make_param('k', 2)
        self.make_param('N', 2*self.r+1)
        self.make_param('width', width)
        self.make_param('height', height)
        self.make_param('rule', rule_value, setter=self.setter_rule)

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
        
        length = self.k**(2*self.r+1)
        self.rule_set = [0]*length
        for i in range(length-1,-1, -1):
            if self.rule >= self.k**i:
                amount_fitted = self.rule // self.k**i
                self.rule -= amount_fitted * self.k**i
                self.rule_set[length-i-1] = amount_fitted
        
        return self.rule_set
    
    def generate_states(self):  
        states = []
        for state in itertools.product(range(self.k), repeat=2*self.r+1):
            states.append(list(state))
        
        states.reverse()
        return states

    def check_rule(self, inp):
        """Returns the new state based on the input states.

        The input state will be an array of 2r+1 items between 0 and k, the
        neighbourhood which the state of the new cell depends on."""
        
        list_states = self.generate_states()
        for i in range(len(list_states)):
            if list_states[i] == list(inp):
                return self.rule_set[i]

                
    def setup_initial_row(self):
        """Returns an array of length `width' with the initial state for each of
        the cells in the first row. Values should be between 0 and k."""
        initial_row = np.zeros(self.width)
        list_k = list(range(self.k))
        list_k.remove(0)
        for i in range(len(initial_row)):
            r = random.random()
            if r > 0.5:
                initial_row[i]= random.choice(list_k)
        return initial_row
    

    def reset(self):
        """Initializes the configuration of the cells and converts the entered
        rule number to a rule set."""

        self.t = 0
        self.config = np.zeros([self.height, self.width])
        self.config[0, :] = self.setup_initial_row()
        self.build_rule_set()

    def draw(self):
        """Draws the current state of the grid."""

        plt.cla()
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.imshow(self.config, interpolation='none', vmin=0, vmax=self.k - 1,
                cmap=plt.cm.binary)
        plt.axis('image')
        plt.title('t = %d' % self.t)

    def step(self):
        """Performs a single step of the simulation by advancing time (and thus
        row) and applying the rule to determine the state of the cells."""
            
        if self.cycle_length == 0:
            self.cycle_l()
    
        if self.t < self.height:
            self.state_history.append(list(self.config[self.t]))

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
    
    def cycle_l(self):
        cycle_list = []
        if self.cycle_length != 0:
            return
        
        for i in range(len(self.state_history)):
            if self.state_history[i] in cycle_list:
                for j in range(len(cycle_list)):
                    if self.state_history[i] == cycle_list[j]:
                        self.cycle_length = i - j
            else:
                cycle_list.append(self.state_history[i])
                
    def langton(self, e_state):
        triangle = self.build_rule_set()
        print(triangle)
        n = 0
        for i in triangle:
            if i == e_state:
                n+=1
        labda = (self.k**self.N - n)/self.k**self.N
        return labda
        
    def random_table(self,e_state,labda):
        rule_set2 = [0]*(self.k**self.N)
        list_k = list(range(self.k))
        list_k.remove(e_state)
        for i in range(len(rule_set2)):
            rand = random.random()
            if rand > labda:
                rule_set2[i] = e_state
            else:
                rule_set2[i] = random.choice(list_k)
        return rule_set2

    def table_walkthrough(self, e_state, labda):
        n = math.floor((self.k**self.N)*(1-labda))
        triangle = [0]*(self.k**self.N)
        list_k = list(range(self.k))
        list_k.remove(e_state)
        position_list = list(range(self.k**self.N))
        for i in range(n):
            x = random.choice(position_list)
            triangle[x]= e_state
            position_list.remove(x)
        for j in position_list: 
            triangle[j]= random.choice(list_k)

            
        return triangle



    def shannon_entropy(self, e_state, labda):
        triangle = self.table_walkthrough(e_state, labda)
        groups = split_into_groups(triangle)
        # Makes it hashable (so we can search the group in the list)
        groups = [tuple(group) for group in groups]
        group_counts = Counter(groups)
        df = pd.DataFrame.from_records(list(group_counts.items()), columns=['Group', 'Count'])
        df['Probability'] = df['Count'] / len(triangle)
        df['Log_Probability'] = np.log(df['Probability'])
        df['Shannon_entropy_i'] = df['Log_Probability'] * df['Probability']
        shannon_entr = -1 * np.sum(df['Shannon_entropy_i'])
        return shannon_entr
     

def get_cycle_length(width=20, height=30, rule_value=213, random_seeds=10):
    cycle_lengths = []
    for random_seed in range(random_seeds):
        sim = CASim(width=width, height=height, rule_value=rule_value, random_seed=random_seed)
        sim.reset()
        
        steps = 0
        while sim.cycle_length==0 and steps<100:
            sim.step()
            steps+=1
        cycle_lengths.append(sim.cycle_length)
        
    return np.mean(cycle_lengths), np.std(cycle_lengths)

def get_transient_length(width=20, height=10000, rule_value=213, random_seeds=10):
    transient_lengths = []
    for random_seed in range(random_seeds):
        sim = CASim(width=width, height=height, rule_value=rule_value, random_seed=random_seed)
        sim.reset()

        steps = 0
        while sim.cycle_length==0 and steps<height:
            sim.step()
            steps+=1
        transient_lengths.append(steps)

    return np.mean(transient_lengths), np.std(transient_lengths)


def split_into_groups(triangle):
    N = len(triangle)
    groups = []
    for i in range(N):
        left = triangle[(i - 1) % N]
        middle = triangle[i]
        right = triangle[(i + 1) % N]
        group = [left, middle, right]
        groups.append(group)
    return groups

def get_rule_number(random_list, k):
    rule = 0
    for i, value in enumerate(random_list):
        rule += value*k**(len(random_list) - i - 1)
    return rule

def get_class_by_rule(df, rule):
    result = df[df['Rule'] == rule]    
    return result.iloc[0]['Class']


df_classes = pd.read_csv('rule_class_wolfram.csv', sep= ',', header=None, names=['Rule', 'Class'])


start_time = time.time()

if __name__ == '__main__':
    sim = CASim()

    lambda_values = np.arange(0, 1.125, 0.125)  # Create an array of lambda values from 0 to 1 in increments of 0.125
    walk_through_table = [sim.table_walkthrough(e_state=1, labda=lambda_value) for lambda_value in lambda_values] 
    random_table_outputs = [sim.random_table(e_state=1, labda=lambda_value) for lambda_value in lambda_values] 

    rules_output_wt = []
    rules_output_rt = []
    class_output_wt = []
    class_output_rt = []
    transient_lengths_wt = []
    transient_lengths_rt = []

    for random_list in walk_through_table:
        rule_number = get_rule_number(random_list, sim.k)
        rules_output_wt.append(rule_number)
    for rule in rules_output_wt:
        rule_class = get_class_by_rule(df_classes, rule)
        class_output_wt.append(rule_class)

    for random_list in random_table_outputs:
        rule_number = get_rule_number(random_list, sim.k)
        rules_output_rt.append(rule_number)
        
    for rule in rules_output_rt:
        rule_class = get_class_by_rule(df_classes, rule)
        class_output_rt.append(rule_class)
    
    for rule in rules_output_rt:
        transient_length = round(get_transient_length(rule_value=rule)[0])
        transient_lengths_rt.append(transient_length)
        
    
    for rule in rules_output_wt:
        transient_length = round(get_transient_length(rule_value=rule)[0])
        transient_lengths_wt.append(transient_length)
    
    
    df_random_table = pd.DataFrame({
        'Lambda': lambda_values,
        'Random table output': random_table_outputs,
        'Rule #': rules_output_rt,
        'Class': class_output_rt,
        'Transient length' : rules_output_rt
    })
    df_walkthrough_table = pd.DataFrame({
        'Lambda': lambda_values,
        'Walk through table': walk_through_table,
        'Rule #': rules_output_wt,
        'Class': class_output_wt,
        'Transient length' : transient_lengths_wt
    })
    
    shannon_outputs = [sim.shannon_entropy(e_state=1, labda=lambda_value) for lambda_value in lambda_values] 
    
    df_shannon = pd.DataFrame({
        'Lambda': lambda_values,
        'Shannon entropies': shannon_outputs
        
    })

    e_state = 1
    # Initialize an empty list to store the entropy values for each run.
    entropy_values = []
    # Lambda values
    labda_values = np.linspace(0, 1, 10000)
    # Run the function 100 times and append the entropy to the list.
    for i in labda_values:
        entropy= sim.shannon_entropy(e_state, labda=i)
        entropy_values.append(entropy)
    
    # Plot the entropy values.
    plt.figure(figsize=(10,6))
    plt.scatter(labda_values, entropy_values, c='blue', s=15, alpha=0.5, edgecolors='r')
    plt.title("Average Single Cell Entropy H Over h Space", fontsize=16)
    plt.xlabel("Lambda", fontsize=14)
    plt.ylabel("Entropy", fontsize=14)
    plt.show()
    """
Shannon entropy measures the unpredictability or randomness of each rule. In the context of Cellular Automata (CAs), 
high Shannon entropy indicates complex, unpredictable patterns, while low Shannon entropy suggests simple, predictable patterns.
The relationship between Shannon entropy and the lambda value is notable, as it shows how bias towards a specific state can impact 
the complexity of patterns produced by the CA. For example, a high lambda value, indicating many configurations leading to a quiescent state, 
would likely generate simple, predictable patterns (low Shannon entropy). Conversely, a low lambda value might create more complex patterns (high Shannon entropy).

The output graph visually depicts the average single cell entropy (H) over h space for approximately 10,000 CA runs. 
Each point represents a different transition function with average H on the y-axis and lambda on the x-axis. 
This graph is similar to Langton's parameter graph, with complexity on the y-axis and Langton's lambda on the x-axis.

Classes 1 and 2, at the lower end of the graph, have lower complexity and lambda values. According to Wolfram's classification, 
these classes evolve into homogenous states or simple, stable structures over time, resulting in shorter, predictable cycle lengths.
 Thus, these classes have less complexity and lower Langton's lambda values, which measure the proportion of non-quiescent states.

Class 3, at the peak of complexity, exhibits the highest complexity with a mid-range lambda value. As per Wolfram's classification, 
Class 3 automata generate seemingly random patterns, leading to longer, less predictable cycle lengths. Hence, the complexity and 
lambda value peak for this class.

Class 4, however, exhibits lower complexity but with the highest lambda values. While this may seem counter-intuitive, 
it aligns with the nature of Class 4 automata. These automata produce complex structures with potential for long-lived, 
computationally universal phenomena. While these structures are diverse and enduring, their overall complexity is less chaotic 
compared to Class 3. Therefore, while the lambda value is high due to the high proportion of non-quiescent states, the overall 
complexity is lower than that of Class 3.
    """



    result = sim.random_table(e_state, labda= 0.25)
    result2 = sim.table_walkthrough(e_state, labda=0.25)
    #result3 = sim.shannon_entropy(e_state=1, labda=0.25)
    results4= get_transient_length()
    
    #print(results4)
    #print(walk_trough_table)
    print(df_walkthrough_table)
    print(df_random_table)
    #print(result)
    #print(result2)
        
end_time = time.time()
elapsed_time = end_time - start_time
#print(f"Function took {elapsed_time} seconds to run.")