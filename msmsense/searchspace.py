import yaml
HP_SPACE = {'feature__value': ['dihedrals', 'distances'],
            'tica__dim': [1, 10], 'tica__lag': [1, 100],
            'cluster__k': [10, 1000], 'cluster__max_iter': [1000],
            'dihedrals__which': ['all'],
            'distances__transform': ['logistic', 'linear'],
            'distances__scheme': ['closest-heavy', 'ca'],
            'distances__steepness': [0.1, 50],
            'distances__centre': [0.3, 1.5]}
#
# with open('searchspace.yaml', 'w') as f:
#     data = yaml.dump(HP_SPACE, f)
#