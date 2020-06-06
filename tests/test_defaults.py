from bioinformatica.defaults import set_default, get_default


def test_data_retrieval():
    key = 'prova'
    value = 42
    set_default(**{key: value})
    assert get_default(key) == value
