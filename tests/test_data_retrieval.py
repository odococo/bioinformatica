from bioinformatica.data_retrieval import data_retrieval


def test_data_retrieval():
    cell_line = 'HepG2'
    region = 'promoters'
    data_retrieval(cell_line=cell_line, region=region)
