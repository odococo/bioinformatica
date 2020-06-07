import os

from bioinformatica.data_analysis import get_tasks
from bioinformatica.data_manipulation import fit_neighbours
from bioinformatica.data_retrieval import data_retrieval, to_bed
from bioinformatica.defaults import set_default


def test_tasks():
    set_default(
        assembly_path=f'{os.getcwd()}/genomes'
    )
    cell_line = 'HEK293'
    en, lab_en = data_retrieval(cell_line, 'enhancers')
    pr, lab_pr = data_retrieval(cell_line, 'promoters')
    en = fit_neighbours(en, 5)
    pr = fit_neighbours(pr, 5)
    epigenomes = {
        'enhancers': en,
        'promoters': pr
    }
    labels = {
        'enhancers': lab_en,
        'promoters': lab_pr
    }
    sequences = {
        'enhancers': to_bed(en),
        'promoters': to_bed(pr)
    }
    _, _, _ = get_tasks(epigenomes, labels, sequences)
