from bioinformatica.data_manipulation import fit_neighbours, drop_too_correlated, show
from bioinformatica.data_retrieval import data_retrieval


def test_data_retrieval():
    cell_line = 'HepG2'
    region = 'promoters'
    epigenomes, labels = data_retrieval(cell_line=cell_line, region=region)
    epigenomes = fit_neighbours(epigenomes, 5)
    scores = drop_too_correlated(epigenomes)
    show({'promoters': epigenomes}, {'promoters': labels}, {'promoters': scores})


if __name__ == '__main__':
    test_data_retrieval()
