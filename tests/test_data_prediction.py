from shutil import rmtree

from bioinformatica.data_analysis import get_filtered_with_boruta
from bioinformatica.data_manipulation import fit_neighbours, apply_z_scoring, drop_constant_features, drop_uncorrelated
from bioinformatica.data_prediction import predict_epigenomics, predict_sequences
from bioinformatica.data_retrieval import data_retrieval, to_bed
from bioinformatica.defaults import set_default, get_default
from bioinformatica.models import get_mlp_epigenomics, get_mlp_sequential


def test_data_prediction():
    set_default(
        cell_line='HEK293',
        region='enhancers',
        epochs=1,
        splits=1,
        batch_size=16
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
        get_mlp_epigenomics()(shape, name="MLP")
    ]
    results_epi = predict_epigenomics(input_data_epi.values, output_data.values.ravel(), epi_models)

    shape = (get_default('window_size'), len(get_default('nucleotides')))
    seq_models = [
        get_mlp_sequential()(shape, name="MLP")
    ]
    results_seq = predict_sequences(input_data_seq, output_data.values.ravel(), seq_models)
    rmtree('datasets')
    rmtree('genome')
    rmtree('epi')
    rmtree('seq')
    rmtree('results_HEK293_enhancers_epi.json')
    rmtree('results_HEK293_enhancers_seq.json')
