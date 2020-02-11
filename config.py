import itertools

# Minimal configuration for testing.
#Best config for m1 and MEE t00
#Run with python train_classification.py --train-file monks/monks-1.train --optimizer t00 --results-folder monks1_res --plots-folder monks1_plots

"""config_options = {
    "l_rate": [0.0005],#0.0001,  0.001, 0.01],
    "num_hidden_neurons": [[20, 20]],
    "weights_distr": ["gaussian"],
    "batch_size": [1], 
    "act_f": ["relu"],#"softplus", "tanh", "logistic"],
    "lambda": [0.001],
    "optimizer": [
        ("adam", 0.9, 0.999, 1e-8),
        ("nesterov", 0.5),
        ("nesterov", 0.9),
        ("trivial", 0.0),  # Same as SGD
        ("trivial", 0.5),
        ("trivial", 0.9)
    ],
}"""

#Best config for m2 and MEE t05
#Run with python train_classification.py --train-file monks/monks-2.train --optimizer t05 --results-folder monks2_res --plots-folder monks2_plots

config_options = {
    "l_rate": [0.05],#0.0001,  0.001, 0.01],
    "num_hidden_neurons": [[20, 20]],
    "weights_distr": ["gaussian"],
    "batch_size": [4], 
    "act_f": ["tanh"],#"softplus", "tanh", "logistic"],
    "lambda": [0.0],
    "optimizer": [
        ("adam", 0.9, 0.999, 1e-8),
        ("nesterov", 0.5),
        ("nesterov", 0.9),
        ("trivial", 0.0),  # Same as SGD
        ("trivial", 0.5),
        ("trivial", 0.9)
    ],
}

#Best config for m3 and MEE t05
#Run with python train_classification.py --train-file monks/monks-3.train --optimizer t05 --results-folder monks3_res --plots-folder monks3_plots

"""config_options = {
    "l_rate": [0.2],#0.0001,  0.001, 0.01],
    "num_hidden_neurons": [150],
    "weights_distr": ["gaussian"],
    "batch_size": [32], 
    "act_f": ["tanh"],#"softplus", "tanh", "logistic"],
    "lambda": [0.0],
    "optimizer": [
        ("adam", 0.9, 0.999, 1e-8),
        ("nesterov", 0.5),
        ("nesterov", 0.9),
        ("trivial", 0.0),  # Same as SGD
        ("trivial", 0.5),
        ("trivial", 0.9)
    ],
}"""



batch_limit = config_options['batch_size'][int((len(config_options['batch_size']) - 1)/2)]

#print(batch_limit)

configs = {
'a': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'adam')
],

't00': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.0)
],

't05': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.5)
],

'asb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'adam') and (v[-4] <= batch_limit)
],

'abb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'adam') and (v[-4] > batch_limit)
],

'n05sb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'nesterov') and (v[-1][1] == 0.5) and (v[-4] <= batch_limit)
],

'n05bb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'nesterov') and (v[-1][1] == 0.5) and (v[-4] > batch_limit)
],

'n09sb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'nesterov') and (v[-1][1] == 0.9) and (v[-4] <= batch_limit)
],

'n09bb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'nesterov') and (v[-1][1] == 0.9) and (v[-4] > batch_limit)
],

't00sb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.0) and (v[-4] <= batch_limit)
],

't00bb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.0) and (v[-4] > batch_limit)
],

't05sb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.5) and (v[-4] <= batch_limit)
],

't05bb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.5) and (v[-4] > batch_limit)
],

't09sb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.9) and (v[-4] <= batch_limit)
],

't09bb': [
    dict(zip(config_options.keys(), v))
    for v in itertools.product(*config_options.values()) if (v[-1][0] == 'trivial') and (v[-1][1] == 0.9) and (v[-4] > batch_limit)
]
}
