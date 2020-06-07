from bioinformatica.data_retrieval import data_retrieval, get_sequences


def test_data_retrieval():
    cell_line = 'HepG2'
    region = 'enhancers'
    input_data, output_data = data_retrieval(cell_line=cell_line, region=region)
    get_sequences(input_data)
