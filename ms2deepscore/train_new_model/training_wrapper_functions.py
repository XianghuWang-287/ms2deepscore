"""Contains wrapper functions that automatically store and load intermediate processed spectra
reducing the amount of rerunning that is necessary"""
import os
from ms2deepscore.train_new_model.SettingMS2Deepscore import \
    SettingsMS2Deepscore
from ms2deepscore.train_new_model.split_positive_and_negative_mode import \
    split_pos_and_neg
from ms2deepscore.train_new_model.train_ms2deepscore import train_ms2ds_model
from ms2deepscore.train_new_model.validation_and_test_split import \
    split_spectra_in_random_inchikey_sets
from ms2deepscore.utils import (load_pickled_file,
                                save_pickled_file)


def train_ms2deepscore_wrapper(data_directory,
                               settings: SettingsMS2Deepscore
                               ):
    trained_models_folder = os.path.join(data_directory, "trained_models")
    os.makedirs(trained_models_folder, exist_ok=True)

    model_folder_file_path = os.path.join(trained_models_folder,
                                          settings.model_directory_name)
    assert not os.path.exists(model_folder_file_path), \
        "The path for this model already exists, choose different settings or remove dir before rerunning"
    os.makedirs(settings.model_directory_name, exist_ok=True)
    # Split training in pos and neg and create val and training split and select for the right ionisation mode.
    training_spectra, validation_spectra = load_train_val_data(data_directory,
                                                               settings.ionisation_mode)
    # Train model
    train_ms2ds_model(training_spectra, validation_spectra, model_folder_file_path, settings)

    return settings.model_directory_name


def store_or_load_neg_pos_spectra(data_directory):
    assert os.path.isdir(data_directory)
    spectra_file_name = os.path.join(data_directory, "cleaned_spectra.mgf")
    assert os.path.isfile(spectra_file_name)

    pos_neg_folder = os.path.join(data_directory, "pos_neg_split")

    # Check if the folder exists otherwise make new folder
    os.makedirs(pos_neg_folder, exist_ok=True)

    positive_mode_spectra_file = os.path.join(pos_neg_folder, "positive_spectra.pickle")
    negative_mode_spectra_file = os.path.join(pos_neg_folder, "negative_spectra.pickle")

    assert os.path.isfile(positive_mode_spectra_file) == os.path.isfile(negative_mode_spectra_file),\
        "One of the pos or neg files was found, both should be there or both should not be there"

    if os.path.isfile(positive_mode_spectra_file):
        positive_mode_spectra = load_pickled_file(positive_mode_spectra_file)
        negative_mode_spectra = load_pickled_file(negative_mode_spectra_file)
        print("Loaded previously stored positive and negative mode spectra")
    else:
        spectra = load_pickled_file(spectra_file_name)
        positive_mode_spectra, negative_mode_spectra = split_pos_and_neg(spectra)
        save_pickled_file(positive_mode_spectra, positive_mode_spectra_file)
        save_pickled_file(negative_mode_spectra, negative_mode_spectra_file)
    return positive_mode_spectra, negative_mode_spectra


def split_or_load_validation_and_test_spectra(data_directory):
    """Will split the spectra based on ionisation mode, unless it is already stored """
    training_and_val_dir = os.path.join(data_directory, "training_and_validation_split")
    os.makedirs(training_and_val_dir, exist_ok=True)

    expected_file_names = [os.path.join(training_and_val_dir, file_name) for file_name in
                           ("positive_validation_spectra.pickle",
                            "positive_training_spectra.pickle",
                            "positive_testing_spectra.pickle",
                            "negative_validation_spectra.pickle",
                            "negative_training_spectra.pickle",
                            "negative_testing_spectra.pickle")]
    files_exist = [os.path.isfile(file_name) for file_name in expected_file_names]
    assert len(set(files_exist)) == 1, "Some of the val, test, train sets exists, but not all"

    if files_exist[0]:
        # Load the files.
        pos_val_spectra, pos_train_spectra, pos_test_spectra, neg_val_spectra, neg_train_spectra, neg_test_spectra = \
            [load_pickled_file(file_name) for file_name in expected_file_names]
        print("Loaded previously stored val, train and test split")
    else:
        positive_spectra, negative_spectra = store_or_load_neg_pos_spectra(data_directory)
        pos_val_spectra, pos_test_spectra, pos_train_spectra = \
            split_spectra_in_random_inchikey_sets(positive_spectra, 20)
        print(f"Positive split \n"
              f"Validation: {len(pos_val_spectra)} \nTrain: {len(pos_train_spectra)} \nTest: {len(pos_test_spectra)}")
        neg_val_spectra, neg_test_spectra, neg_train_spectra = \
            split_spectra_in_random_inchikey_sets(negative_spectra, 20)
        print(f"Negative split \n"
              f"Validation: {len(neg_val_spectra)} \nTrain: {len(neg_train_spectra)} \nTest: {len(neg_test_spectra)}")
        for i, spectra_to_store in enumerate((pos_val_spectra, pos_train_spectra, pos_test_spectra,
                                              neg_val_spectra, neg_train_spectra, neg_test_spectra)):
            save_pickled_file(spectra_to_store, expected_file_names[i])
    return pos_val_spectra, pos_train_spectra, pos_test_spectra, neg_val_spectra, neg_train_spectra, neg_test_spectra


def load_train_val_data(data_directory, ionisation_mode):
    assert ionisation_mode in ("positive", "negative", "both")
    pos_val_spectra, pos_train_spectra, _, neg_val_spectra, neg_train_spectra, _ = \
        split_or_load_validation_and_test_spectra(data_directory)
    if ionisation_mode == "positive":
        return pos_train_spectra, pos_val_spectra
    if ionisation_mode == "negative":
        return neg_train_spectra, neg_val_spectra
    if ionisation_mode == "both":
        training_spectra = pos_train_spectra + neg_train_spectra
        validatation_spectra = pos_val_spectra + neg_val_spectra
        return training_spectra, validatation_spectra
    return None, None
