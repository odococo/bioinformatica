from bioinformatica.data_analysis import show_decomposed_data
from bioinformatica.data_manipulation import fit_neighbours
from bioinformatica.data_retrieval import data_retrieval


def test_data_retrieval():
    cell_line = 'HepG2'
    region = 'promoters'
    epigenomes, labels = data_retrieval(cell_line=cell_line, region=region)
    epigenomes = fit_neighbours(epigenomes, 5)
    xs = [*[epigenomes.values]]
    ys = [*[labels.values.ravel()]]
    titles = ['Epigenomes promoters']
    show_decomposed_data(xs, ys, titles)
