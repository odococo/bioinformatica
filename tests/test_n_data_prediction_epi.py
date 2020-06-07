import os

from bioinformatica.data_analysis import get_filtered_with_boruta
from bioinformatica.data_manipulation import fit_neighbours, apply_z_scoring, drop_constant_features, drop_uncorrelated
from bioinformatica.data_prediction import predict_epigenomics, show_barplots, predict_sequences
from bioinformatica.data_retrieval import data_retrieval, to_bed
from bioinformatica.defaults import set_default, get_default
from bioinformatica.models import get_mlp_epigenomics, get_ffnn_epigenomics_v1, get_mlp_sequential


def test_data_prediction():
    set_default(
        cell_line='HEK293',
        region='enhancers',
        epochs=1,
        splits=2,
        batch_size=1024,
        boruta_iterations=2,
        results_path=f'{os.getcwd()}/results'
    )
    input_data_o, output_data = data_retrieval(get_default('cell_line'), get_default('region'))
    input_data_seq = to_bed(input_data_o)
    input_data_epi = fit_neighbours(input_data_o, 5)
    input_data_epi = apply_z_scoring(input_data_epi)
    input_data_epi = drop_constant_features(get_default('region'), input_data_epi)
    input_data_epi = drop_uncorrelated(input_data_epi, output_data)
    input_data_epi = get_filtered_with_boruta(input_data_epi, output_data,
                                              get_default('cell_line'),
                                              get_default('region'))
    shape = (input_data_epi.shape[1],)
    epi_models = [
        get_mlp_epigenomics()(shape, validation_split=0.1, name="MLP1"),
        get_ffnn_epigenomics_v1()(shape, validation_split=0.1, name="MLP2")
    ]
    results = predict_epigenomics(input_data_epi.values, output_data.values.ravel(), epi_models)
    show_barplots(results, 'epi')
    shape = (get_default('window_size'), len(get_default('nucleotides')))
    seq_models = [
        get_mlp_sequential()(shape, batch_size=None, name="MLP")
    ]
    results = predict_sequences(input_data_seq, output_data.values.ravel(), seq_models)
    show_barplots(results, 'seq')
