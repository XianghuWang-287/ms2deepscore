import os
import sys
import numpy as np
from matchms.importing import load_from_json
import tensorflow as tf
import pickle
from rdkit import Chem
from collections import Counter
import pandas as pd
from matchms.filtering.add_fingerprint import add_fingerprint
from matchms.similarity import FingerprintSimilarity
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
import matplotlib.pyplot as plt


def smiles_to_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchi = Chem.MolToInchi(mol)
    inchikey = Chem.InchiToInchiKey(inchi)
    return inchi,inchikey

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def count_annotations(spectra):
    inchi_lst = []
    smiles_lst = []
    inchikey_lst = []
    for i, spec in enumerate(spectra):
        inchi_lst.append(spec.get("inchi"))
        smiles_lst.append(spec.get("smiles"))
        inchikey = spec.get("inchikey")
        if inchikey is None:
            inchikey = spec.get("inchikey_inchi")
        inchikey_lst.append(inchikey)

    inchi_count = sum([1 for x in inchi_lst if x])
    smiles_count = sum([1 for x in smiles_lst if x])
    inchikey_count = sum([1 for x in inchikey_lst if x])
    print(f"Inchis: {inchi_count} -- {len(set(inchi_lst))} unique")
    print(f"Smiles: {smiles_count} -- {len(set(smiles_lst))} unique")
    print("Inchikeys:", inchikey_count, "--",
          len(set(inchikey_lst)), "unique")
    print("Inchikeys:", inchikey_count, "--",
          len(set([x[:14] for x in inchikey_lst if x])), "unique (first 14 characters)")

def annotated(s):
    return (s.get("inchi") or s.get("smiles")) and s.get("inchikey")


def precision_recall_plot(scores_test, scores_ref,
                          high_sim_threshold=0.6,
                          n_bins=20):
    """Basic precision recall plot"""
    precisions = []
    recalls = []

    above_thres_total = np.sum(scores_ref >= high_sim_threshold)
    max_score = scores_test.max()
    min_score = scores_test.min()
    score_thresholds = np.linspace(min_score, max_score, n_bins + 1)
    for low in score_thresholds:
        idx = np.where(scores_test >= low)
        above_thres = np.sum(scores_ref[idx] >= high_sim_threshold)
        below_thres = np.sum(scores_ref[idx] < high_sim_threshold)

        precisions.append(above_thres / (below_thres + above_thres))
        recalls.append(above_thres / above_thres_total)

    plt.figure(figsize=(6, 5), dpi=120)
    plt.plot(recalls, precisions, "o--", color="crimson", label="precision/recall")
    # plt.plot(score_thresholds, precisions, "o--", color="crimson", label="precision")
    # plt.plot(score_thresholds, recalls, "o--", color="dodgerblue", label="recall")
    plt.legend()
    plt.xlabel("recall", fontsize=12)
    plt.ylabel("precision", fontsize=12)
    plt.title(f"precision/recall (high-similarity if Tanimoto > {high_sim_threshold})")
    plt.grid(True)

    return precisions, recalls


# %%
def tanimoto_dependent_losses(scores, scores_ref, ref_score_bins):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).

    Parameters
    ----------

    scores
        Scores that should be evaluated
    scores_ref
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the refernce score to evaluate the performance of scores.
    """
    bin_content = []
    rmses = []
    maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    for i in range(len(ref_scores_bins_inclusive) - 1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i + 1]
        bounds.append((low, high))
        idx = np.where((scores_ref >= low) & (scores_ref < high))
        bin_content.append(idx[0].shape[0])
        maes.append(np.abs(scores_ref[idx] - scores[idx]).mean())
        rmses.append(np.sqrt(np.square(scores_ref[idx] - scores[idx]).mean()))

    return bin_content, bounds, rmses, maes


def plot_tanimoto_dependent_losses(bin_content, bounds, rmses, maes):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 8), dpi=120)
    ax1.plot(np.arange(len(rmses)), maes, "o--")
    ax1.set_title('MAE')
    ax1.set_ylabel("MAE")
    # ax1.set_ylim(0)
    ax1.grid(True)

    ax2.plot(np.arange(len(rmses)), rmses, "o--", color="crimson")
    ax2.set_title('RMSE')
    ax2.set_ylabel("RMSE")
    # ax2.set_xlabel("score STD threshold")
    # ax2.set_ylim(0)
    ax2.grid(True)

    ax3.plot(np.arange(len(rmses)), bin_content, "o--", color="teal")
    ax3.set_title('# of spectrum pairs')
    ax3.set_ylabel("# of spectrum pairs")
    ax3.set_xlabel("Tanimoto score bin")
    plt.xticks(np.arange(len(rmses)),
               [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    # ax3.set_ylim(0)
    ax3.grid(True)


def tanimoto_dependent_STDs(scores_STD, scores_ref, bins):
    bin_content = []
    STDs = []
    for i in range(len(ref_score_bins) - 1):
        low = ref_score_bins[i]
        high = ref_score_bins[i + 1]
        idx = np.where((scores_ref >= low) & (scores_ref < high))
        bin_content.append(idx[0].shape[0])
        STDs.append(scores_STD[idx].mean())
    return bin_content, STDs


ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, ROOT)


outfile = os.path.join(ROOT,'ms2deepscore','GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_processed.pickle')
with open(outfile, 'rb') as file:
    spectrums = pickle.load(file)

for spec in spectrums:
    smiles = spec.get('smiles')
    inchi, inchikey = smiles_to_inchikey(smiles)
    spec.set('inchi',inchi)
    spec.set('inchikey',inchikey)
    print(spec.metadata['inchikey'])

count_annotations(spectrums)

annotation_list = []
for i, s in enumerate(spectrums):
    if annotated(s):
        annotation_list.append((i, s.get("inchi"), s.get("smiles"), s.get("inchikey")))

spectrums_annotated = [s for s in spectrums if annotated(s)]

inchikeys_list = []
for s in spectrums_annotated:
    inchikeys_list.append(s.get("inchikey"))

inchikeys14_array = np.array([x[:14] for x in inchikeys_list])
inchikeys14_unique = list({x[:14] for x in inchikeys_list})

inchi_list = []
for s in spectrums_annotated:
    inchi_list.append(s.get("inchi"))

inchi_array = np.array(inchi_list)

inchi_mapping = []
ID_mapping = []

for inchikey14 in inchikeys14_unique:
    idx = np.where(inchikeys14_array == inchikey14)[0]

    inchi = most_frequent([spectrums_annotated[i].get("inchi") for i in idx])
    inchi_mapping.append(inchi)
    ID = idx[np.where(inchi_array[idx] == inchi)[0][0]]
    ID_mapping.append(ID)

metadata = pd.DataFrame(list(zip(inchikeys14_unique, inchi_mapping, ID_mapping)), columns=["inchikey", "inchi", "ID"])



for i in metadata.ID.values:
    spectrums_annotated[i] = add_fingerprint(spectrums_annotated[i],fingerprint_type="daylight", nbits=2048)

spectrums_represent = [spectrums_annotated[i] for i in metadata.ID.values]

similarity_measure = FingerprintSimilarity(similarity_measure="jaccard")
scores_mol_similarity = similarity_measure.matrix(spectrums_represent, spectrums_represent)

tanimoto_df = pd.DataFrame(scores_mol_similarity, columns=metadata.inchikey.values, index=metadata.inchikey.values)

#test the pre-trained
model_pre = load_model("MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5")

similarity_score = MS2DeepScore(model_pre)
similarities_test = similarity_score.matrix(spectrums_represent, spectrums_represent, is_symmetric=True)

inchikey_idx_test = np.zeros(len(spectrums_represent))
for i, spec in enumerate(spectrums_represent):
    inchikey_idx_test[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0]

inchikey_idx_test = inchikey_idx_test.astype("int")

scores_ref = tanimoto_df.values[np.ix_(inchikey_idx_test[:], inchikey_idx_test[:])].copy()

ref_score_bins = np.linspace(0,1.0, 11)
bin_content_pre, bounds_pre, rmses_pre, maes_pre = tanimoto_dependent_losses(similarities_test,
                                                             scores_ref, ref_score_bins)
#test the retrained
model =load_model("ms2deepscore_model.hdf5")
similarity_score = MS2DeepScore(model)
similarities_test = similarity_score.matrix(spectrums_represent, spectrums_represent, is_symmetric=True)
bin_content, bounds, rmses, maes = tanimoto_dependent_losses(similarities_test,
                                                             scores_ref, ref_score_bins)


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 5), dpi=120)

ax1.plot(np.arange(len(rmses)), rmses, "o:", color="b",label = 're-trained')
ax1.plot(np.arange(len(rmses_pre)), rmses_pre, "o:", color="crimson",label = 'pre-trained')
ax1.set_title('RMSE')
ax1.set_ylabel("RMSE")
ax1.legend()
ax1.grid(True)

ax2.plot(np.arange(len(rmses)), bin_content, "o:", color="teal")
ax2.set_title('# of spectrum pairs')
ax2.set_ylabel("# of spectrum pairs")
ax2.set_xlabel("Tanimoto score bin")
plt.yscale('log')
plt.xticks(np.arange(len(rmses)),
           [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
ax2.grid(True)
plt.tight_layout()
plt.show()





