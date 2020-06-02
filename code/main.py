from data_manipulation import fit_neighbours, apply_z_scoring
from data_prediction import predict_epigenomics, predict_sequences, show_barplots
from data_retrieval import data_retrieval, get_sequences
from defaults import set_default, get_default
from models import get_mlp_egigenomics, get_ffnn_epigenomics_v1, get_ffnn_epigenomics_v2, get_ffnn_epigenomics_v3, \
    get_mlp_sequential, get_ffnn_sequential, get_cnn_sequential_v1

cell_line = 'HEK293'
region = 'enhancers'

set_default(
    assembly='hg19'  # path
)

if __name__ == '__main__':
    input_data, output_data = data_retrieval(cell_line, region)
    input_data = fit_neighbours(input_data, input_data.shape[0] // 10)
    input_data = apply_z_scoring(input_data)

    shape = (input_data.shape[1],)
    epi_models = [
        get_mlp_egigenomics()(shape, name="MLP"),
        get_ffnn_epigenomics_v1()(shape, name="FFNN_1"),
        get_ffnn_epigenomics_v2()(shape, name="FFNN_2"),
        get_ffnn_epigenomics_v3()(shape, name="FFNN_3")
    ]
    results = predict_epigenomics(input_data, output_data, epi_models)
    show_barplots(results)

    input_data = get_sequences(input_data)

    shape = (get_default('window_size'), get_default('nucleotides'))
    seq_models = [
        get_mlp_sequential()(shape, name="MLP"),
        get_ffnn_sequential()(shape, name="FFNN"),
        get_cnn_sequential_v1()(shape, name="CNN_1")
    ]
    results = predict_sequences(input_data, output_data, seq_models)
    show_barplots(results)
