def test_package_imports():
    import maggpy

    assert maggpy.__version__ == "0.1.0"

def test_core_modules_import():
    import maggpy.data_io
    import maggpy.redshift
    import maggpy.spectral_models
    import maggpy.top_hat.montecarlo