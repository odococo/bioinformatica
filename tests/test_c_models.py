from bioinformatica.meta_models import Models
from bioinformatica.models import get_mlp_epigenomics, get_ffnn_epigenomics_v1, get_ffnn_epigenomics_v2, \
    get_ffnn_epigenomics_v3, get_mlp_sequential, get_ffnn_sequential, get_cnn_sequential_v1


def test_models():
    get_mlp_epigenomics()
    get_ffnn_epigenomics_v1()
    get_ffnn_epigenomics_v2()
    get_ffnn_epigenomics_v3()
    get_mlp_sequential()
    get_ffnn_sequential()
    get_cnn_sequential_v1()
    shape = (200, 4)
    model = Models.Perceptron(shape)
    str(model)
    repr(model)
    Models.DecisionTree()
    Models.RandomForest()
