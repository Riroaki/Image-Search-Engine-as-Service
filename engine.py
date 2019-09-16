import logging
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from lshash.lshash import LSHash
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger('Image search:engine')
logger.setLevel(logging.DEBUG)
# Use ORB to extract feature vectors
orb = cv2.ORB_create()


class ImageSearchEngine(object):
    """A simple image search engine based on ORB, Kmeans and LSHash."""

    def __init__(self):
        self._all_feats = []
        self._img_dict = {}
        self._kmeans = None
        self._lsh = None

    def load_images(self, img_list: list) -> int:
        """Load images, extract features using ORB for indexing.

        Args:
            img_list: list of image files' names.

        Returns:
            count of image files successfully loaded.
        """
        count = 0
        progress_bar = tqdm(total=len(img_list))
        for img_name in img_list:
            try:
                img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
                _, features = orb.detectAndCompute(img, None)
                # Record index of features for this image
                start_index, num_feats = len(self._all_feats), len(features)
                self._img_dict[img_name] = {'start_index': start_index,
                                            'num_feats': num_feats}
                # Append new featurs
                self._all_feats.extend([feat for feat in features])
                count += 1
            except Exception as e:
                logger.warning(e)
                logger.warning('Error processing {}'.format(img_name))
            progress_bar.update(1)
        progress_bar.close()
        logger.info('Successfully loaded {} images, extracted {} features.'
                    .format(count, len(self._all_feats)))
        return count

    def build_index(self, k: int, hash_size: int = 10, num_hashtables: int = 1,
                    store_file: str = None, overwrite: bool = False):
        """Build index for each picture.
        First use K-means to find k key features from previously extracted features and the assignment of each feature;

        Then apply histogram on each image, get the distribution of its features, which serves as a unique finger print for this image.

        Finally use LSHash (locality sensitive hashing.) algorithm, index each image by their histogram array.

        Args:
            k: parameter used in K-means, number of centeroids (key features).
            hash_size: length of resulting binary hash array.
            num_hashtables: number of hashtables for multiple lookups.
            store_file: Specify the path to the .npz file random matrices are stored or to be stored if the file does not exist yet
            overwrite: Whether to overwrite the matrices file if it already exist.

        Returns:

        """
        assert 0 < k < len(self._all_feats)
        assert hash_size > 0 and num_hashtables > 0

        # Use kmeans to calculate K key features and assignment of each feature.
        logger.info('Calculating {} key featurs...'.format(k))
        # Mini batch kmeans deals with large amount of data better.
        self._kmeans = MiniBatchKMeans(n_clusters=k)
        self._kmeans.fit(np.array(self._all_feats))
        idx = self._kmeans.labels_
        logger.info('Start indexing each image.')

        # Calculate histogram of each image
        self._lsh = LSHash(hash_size=hash_size,
                           input_dim=k,
                           num_hashtables=num_hashtables,
                           matrices_filename=store_file,
                           overwrite=overwrite)
        success = 0
        progress_bar = tqdm(total=len(self._img_dict))
        bins = np.arange(-0.5, k + 0.5, 1)
        for img_name, img_meta in self._img_dict.items():
            try:
                start = img_meta['start_index']
                end = start + img_meta['num_feats']
                # Perform histogram
                hist, _ = np.histogram(idx[start:end], bins=bins)
                img_meta['histogram'] = hist
                # Store each picture in hash tables
                self._lsh.index(input_point=hist, extra_data=img_name)
                success += 1
            except Exception as e:
                logger.warning(e)
                logger.warning('Error when indexing image: {}'.format(img_name))
            progress_bar.update(1)
        progress_bar.close()
        logger.info('Successfully indexed {} images.'.format(success))

    def search(self, img_name: str, num_results: int = None,
               distance_func: str = None) -> list:
        """Search image.

        Args:
            img_name: name of image file to searched.
            num_results: The number of query results to return in ranked order. By default all results will be returned.
            distance_func: The distance function to be used, in ("hamming", "euclidean", "true_euclidean", "centred_euclidean", "cosine", "l1norm").
                By default "euclidean" will used.

        Returns:
            list of names of match images.
        """
        assert self._lsh is not None and self._kmeans is not None
        res = []
        try:
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            _, features = orb.detectAndCompute(img, None)
            idx = self._kmeans.predict(features)
            bins = np.arange(-0.5, len(self._kmeans.cluster_centers_) + 0.5, 1)
            hist, _ = np.histogram(idx, bins=bins)
            res = self._lsh.query(hist, num_results=num_results,
                                  distance_func=distance_func)
        except Exception as e:
            logger.warning(e)
        return res

    def dump(self, pkl_file: str = 'model.pkl'):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self, f)

    @property
    def num_images(self) -> int:
        return len(self._img_dict)
