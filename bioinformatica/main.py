from .data_analysis import get_filtered_with_boruta
from .data_manipulation import fit_neighbours, apply_z_scoring, drop_constant_features, drop_uncorrelated
from .data_prediction import predict_epigenomics, predict_sequences, show_barplots
from .data_retrieval import data_retrieval, to_bed
from .defaults import set_default, get_default
from .models import get_mlp_epigenomics, get_ffnn_epigenomics_v1, get_ffnn_epigenomics_v2, get_ffnn_epigenomics_v3, \
    get_mlp_sequential, get_ffnn_sequential, get_cnn_sequential_v1

set_default(
    assembly='hg19',  # path
    cell_line='HEK293',
    region='promoters',
    dataset_path=r'C:\Users\matte\Documents\GitHub\bioinformatica\HepG2\datasets'

)

if __name__ == '__main__':
    input_data_o, output_data = data_retrieval(get_default('cell_line'), get_default('region'))

    input_data_seq = to_bed(input_data_o)  # annotate genome using index extracted from epigenomic data

    # epigenomic data's preproceccing
    input_data_epi = fit_neighbours(input_data_o, 5)  # NaN imputation
    input_data_epi = apply_z_scoring(input_data_epi)  # Normalizing
    # feature selection
    #input_data_epi = drop_constant_features(get_default('region'), input_data_epi)
    #input_data_epi = drop_uncorrelated(input_data_epi, output_data)
    #input_data_epi = get_filtered_with_boruta(input_data_epi, output_data, get_default('cell_line'),get_default('region'))

    shape = (input_data_epi.shape[1],)
    epi_models = [
        get_mlp_epigenomics()(shape, name="MLP"),
        get_ffnn_epigenomics_v1()(shape, name="FFNN_1"),
        get_ffnn_epigenomics_v2()(shape, name="FFNN_2"),
        get_ffnn_epigenomics_v3()(shape, name="FFNN_3")
    ]
    results_epi = predict_epigenomics(input_data_epi.values, output_data.values.ravel(), epi_models)
    show_barplots(results_epi, "epi")

    shape = (get_default('window_size'), len(get_default('nucleotides')))
    seq_models = [
        get_mlp_sequential()(shape, name="MLP"),
        get_ffnn_sequential()(shape, name="FFNN"),
        get_cnn_sequential_v1()(shape, name="CNN_1")
    ]
    results_seq = predict_sequences(input_data_seq, output_data.values.ravel(), seq_models)
    show_barplots(results_seq, "seq")
