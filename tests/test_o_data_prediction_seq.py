import os

from bioinformatica.data_prediction import show_barplots, predict_sequences
from bioinformatica.data_retrieval import data_retrieval, to_bed
from bioinformatica.defaults import set_default, get_default
from bioinformatica.models import get_mlp_sequential


def test_data_prediction():
    set_default(
        cell_line='HepG2',
        region='promoters',
        epochs=4,
        splits=2,
        batch_size=1024,
        boruta_iterations=2,
        results_path=f'{os.getcwd()}/results',
        assembly_path=f'{os.getcwd()}/genomes'
    )
    input_data, output_data = data_retrieval(get_default('cell_line'), get_default('region'))
    input_data_seq = to_bed(input_data)
    shape = (get_default('window_size'), len(get_default('nucleotides')))
    seq_models = [
        get_mlp_sequential()(shape, name="MLP")
    ]
    results = predict_sequences(input_data_seq, output_data.values.ravel(), seq_models)
    show_barplots(results, 'seq')


if __name__ == '__main__':
    test_data_prediction()
