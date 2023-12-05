import os
import pickle

from matchms.exporting import save_as_mgf
from tests.create_test_spectra import pesticides_test_spectra

from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.wrapper_functions.training_wrapper_functions import \
    train_ms2deepscore_wrapper
from ms2deepscore.wrapper_functions.StoreTrainingData import StoreTrainingData


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def test_train_wrapper_ms2ds_model(tmp_path):
    spectra = pesticides_test_spectra()
    positive_mode_spectra = [spectrum.set("ionmode", "positive") for spectrum in spectra[:35]]
    negative_mode_spectra = [spectrum.set("ionmode", "negative") for spectrum in spectra[35:]]

    save_as_mgf(positive_mode_spectra+negative_mode_spectra,
                filename=os.path.join(tmp_path, "clean_spectra.mgf"))
    settings = SettingsMS2Deepscore({"epochs": 2,
                                     "average_pairs_per_bin": 2,
                                     "ionisation_mode": "negative",
                                     "batch_size": 2})
    train_ms2deepscore_wrapper(tmp_path, "clean_spectra.mgf", settings, 5)
    expected_file_names = StoreTrainingData(tmp_path, "clean_spectra.mgf")
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, settings.model_file_name))
    assert os.path.isfile(expected_file_names.negative_mode_spectra_file)
    assert os.path.isfile(expected_file_names.negative_validation_spectra_file)
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, "benchmarking_results",
                                       "both_both_predictions.pickle"))
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, "benchmarking_results",
                                       "plots", "both_both_plot.svg"))
    assert os.path.isfile(os.path.join(tmp_path, expected_file_names.trained_models_folder,
                                       settings.model_directory_name, "binned_spectra",
                                       "binned_training_spectra.pickle"))

