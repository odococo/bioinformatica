import json

import pandas as pd

from bioinformatica.data_prediction import t_wilcoxon


def test_wilcoxon():
    with open('results/results_HEK293_enhancers_epi.json') as json_file:
        results = json.load(json_file)

    df = pd.DataFrame(results)

    models = df[
        (df.run_type == "test")
    ]
    print(models)
    print(models[models.model == 'MLP'])
    print(models[models.model == 'Perceptron'])
    t_wilcoxon(models[models.model == 'MLP'], models[models.model == 'Perceptron'])


if __name__ == '__main__':
    test_wilcoxon()
