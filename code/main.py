from data_manipulation import fit_neighbours, apply_z_scoring
from data_prediction import predict_epigenomics
from data_retrieval import data_retrieval, set_default
from models import get_mlp_egigenomics, get_ffnn_epigenomics_v1, get_ffnn_epigenomics_v2, get_ffnn_epigenomics_v3

cell_line = 'HEK293'
regions = 'enhancers'

set_default(
    assembly='hg19'  # path
)

if __name__ == '__main__':
    epigenomes, labels = data_retrieval(cell_line)

    input_data = {}

    for region, data in epigenomes.items():
        data_imputed = fit_neighbours(data, data.shape[0] // 10)
        data_normalized = apply_z_scoring(data_imputed)
        input_data[region] = data_normalized

    data = input_data[regions]
    output = labels[regions]

    shape = (data.shape[1],)

    models = []
    models.append(get_mlp_egigenomics()(shape, name="MLP"))
    models.append(get_ffnn_epigenomics_v1()(shape, name="FFNN_1"))
    models.append(get_ffnn_epigenomics_v2()(shape, name="FFNN_2"))
    models.append(get_ffnn_epigenomics_v3()(shape, name="FFNN_3"))

    results = predict_epigenomics(data, output)
