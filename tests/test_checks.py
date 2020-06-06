from shutil import rmtree

from bioinformatica.data_manipulation import overfitting_risk, nan_check, check_class_balance
from bioinformatica.data_retrieval import data_retrieval


def test_checks():
    cell_line = 'HEK293'
    en, lab_en = data_retrieval(cell_line, 'enhancers')
    pr, lab_pr = data_retrieval(cell_line, 'promoters')
    epigenomes = {
        'enhancers': en,
        'promoters': pr
    }
    labels = {
        'enhancers': lab_en,
        'promoters': lab_pr
    }
    overfitting_risk(epigenomes)
    nan_check(epigenomes)
    check_class_balance(labels)

    rmtree('datasets')


if __name__ == '__main__':
    test_checks()
