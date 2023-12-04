import string
import numpy as np
from ms2deepscore.visualize_results.calculate_tanimoto_scores import calculate_tanimoto_scores_unique_inchikey, \
    get_tanimoto_score_between_spectra
from pathlib import Path
from matchms import Spectrum

TEST_RESOURCES_PATH = Path(__file__).parent / 'resources'


def create_dummy_data(nr_of_spectra):
    """Create fake data to test generators.
    """
    # Create fake smiles
    spectrums = []
    for i in range(nr_of_spectra):
        smiles = "C" * (i+1)
        dummy_inchikey = f"{14 * string.ascii_uppercase[i]}-{10 * string.ascii_uppercase[i]}-N"
        spectrum = Spectrum(mz=np.array([100.0 + (i+1) * 25.0]), intensities=np.array([0.1]),
                            metadata={"inchikey": dummy_inchikey,
                                      "smiles": smiles})
        spectrums.append(spectrum)
    return spectrums


def test_get_tanimoto_score_between_spectra():
    nr_of_test_spectra = 3
    spectrums = create_dummy_data(nr_of_test_spectra)
    tanimoto_scores = get_tanimoto_score_between_spectra(spectrums+spectrums,
                                                         spectrums)
    assert tanimoto_scores.shape == (nr_of_test_spectra*2, nr_of_test_spectra)
    expected_values = np.array([[0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.5],
                                [0.0, 0.5, 1.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.5],
                                [0.0, 0.5, 1.0],
                                ])
    assert np.array_equal(tanimoto_scores, expected_values)


def test_calculate_tanimoto_scores_unique_inchikey():
    nr_of_test_spectra = 4
    spectrums = create_dummy_data(nr_of_test_spectra)
    tanimoto_scores = calculate_tanimoto_scores_unique_inchikey(
        spectrums + spectrums,
        spectrums)
    assert tanimoto_scores.shape == (nr_of_test_spectra, nr_of_test_spectra)
