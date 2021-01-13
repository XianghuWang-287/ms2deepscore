from typing import List
from typing import Union
import numpy
from matchms.similarity.BaseSimilarity import BaseSimilarity
from ms2deepscore.vector_operations import cosine_similarity
from ms2deepscore.vector_operations import cosine_similarity_matrix
from tqdm import tqdm
from ms2deepscore import BinnedSpectrum
from ms2deepscore import SpectrumBinner
from ms2deepscore import SiameseModel


class MS2DeepScore(BaseSimilarity):
    """Calculate MS2DeepScore similarity scores between a reference and a query.

    Using a trained model, binned spectrums will be converted into spectrum
    vectors using a deep neural network. The MS2DeepScore similarity is then
    the cosine similarity score between two spectrum vectors.

    Example code to calcualte MS2DeepScore similarities between query and reference
    spectrums:

    .. code-block:: python

        from matchms import calculate_scores
        from ms2deepscore import SpectrumBinner
        from ms2deepscore import SiameseModel

        # reference_spectrums & query_spectrums loaded from files using https://matchms.readthedocs.io/en/latest/api/matchms.importing.load_from_mgf.html
        ms2ds_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
        references = ms2ds_binner.fit_transform(reference_spectrums)
        ...
        
    """
    def __init__(self, model, progress_bar: bool = False):
        """

        Parameters
        ----------
        model:
            Expected input is a SiameseModel that has been trained on
            the desired set of spectra.
        intensity_weighting_power:
            Spectrum vectors are a weighted sum of the word vectors. The given
            word intensities will be raised to the given power.
            The default is 0, which means that no weighing will be done.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.vector_size = self.model.base.output_shape[1]
        self.disable_progress_bar = not progress_bar

    def pair(self, reference: BinnedSpectrum, query: BinnedSpectrum) -> float:
        """Calculate the MS2DeepScore similaritiy between a reference and a query.

        Parameters
        ----------
        reference:
            Reference binned spectrum.
        query:
            Query binned spectrum.

        Returns
        -------
        ms2ds_similarity
            MS2DeepScore similarity score.
        """
        reference_vector = self.model.base.predict(reference)
        query_vector = self.model.base.predict(query)

        return cosine_similarity(reference_vector, query_vector)

    def matrix(self, references: List[SpectrumDocument], queries: List[SpectrumDocument],
               is_symmetric: bool = False) -> numpy.ndarray:
        """Calculate the MS2DeepScore similarities between all references and queries.

        Parameters
        ----------
        references:
            Reference spectrum documents.
        queries:
            Query spectrum documents.
        is_symmetric:
            Set to True if references == queries to speed up calculation about 2x.
            Uses the fact that in this case score[i, j] = score[j, i]. Default is False.

        Returns
        -------
        ms2ds_similarity
            Array of MS2DeepScore similarity scores.
        """
        n_rows = len(references)
        reference_vectors = numpy.empty((n_rows, self.vector_size), dtype="float")
        for index_reference, reference in enumerate(tqdm(references, desc='Calculating vectors of reference spectrums', disable=self.disable_progress_bar)):
            reference_vectors[index_reference, 0:self.vector_size] = self.model.base.predict(reference)
        n_cols = len(queries)
        query_vectors = numpy.empty((n_cols, self.vector_size), dtype="float")
        for index_query, query in enumerate(tqdm(queries, desc='Calculating vectors of query spectrums', disable=self.disable_progress_bar)):
            query_vectors[index_query, 0:self.vector_size] = self.model.base.predict(query))

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)

        return ms2ds_similarity
