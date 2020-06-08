import json

import pandas as pd

from bioinformatica.data_prediction import t_wilcoxon


def test_wilcoxon():
    with open('results/results_wilcoxon.json') as json_file:
        results = json.load(json_file)

    df = pd.DataFrame(results)

    models = df[
        (df.run_type == "test")
    ]
    t_wilcoxon(models[models.model == 'MLP'], models[models.model == 'FFNN'])
