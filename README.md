# Image Search Engine as Service

> A simple image search engine based on ORB, LSHash and K-Means algorithm.
>
> Deployment enabled using Flask.

## Requirements

- Python3
- numpy==1.17.0
- opencv-python==4.1.1.26
- scikit-learn==0.20.3
- lshash3==0.0.8
- flask==1.1.1

## Process

### Build index of source images

- Use ORB to extract feature points and corresponding feature vectors from each image.
- Apply (mini-batch) K-Means algorithm, aggregate feature vectors and find K centeroids (key features).
- Use histogram to describe each images' features, that is, calculate the assignments of all feature vectors of each image, and use histogram vectors of assignments to represent the signatures of each image.
- Finally, put the histogram vector into LSHash tables.

### Search images using POST requests

| Name       | Type | Optional | Default     | Meaning                                            |
| ---------- | ---- | -------- | ----------- | -------------------------------------------------- |
| `image`    | str  | false    | /           | Path of your image file.                           |
| `n`        | int  | true     | None        | Number of search results, default means unlimited. |
| `distance` | str  | true     | 'Euclidean' | Name of distance function used in LSHash.          |

## Usage

```shell
$ python service.py -h
usage: Deploy image search engine as a local service. [-h] [--host HOST]
                                                      [--port PORT] --data
                                                      DATA --k K
                                                      [--hash_size HASH_SIZE]
                                                      [--num_hashtables NUM_HASHTABLES]
                                                      [--store_file STORE_FILE]
                                                      [--overwrite OVERWRITE]

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           host of your image search service
  --port PORT, -p PORT  port of your image search service
  --data DATA, -d DATA  image folder which contains source images
  --k K, -k K           number of key features used in K-Means
  --hash_size HASH_SIZE
                        length of resulting binary hash array
  --num_hashtables NUM_HASHTABLES
                        number of hashtables for multiple lookups.
  --store_file STORE_FILE, -s STORE_FILE
                        the path to the .npz file random matrices are stored
  --overwrite OVERWRITE, -o OVERWRITE
                        whether to overwrite the matrices file if exist
# sample use:
$ python service.py --data data --k 100 --hash_size 11 --num_hashtables 3
```

## About algorithms applied in this project

- ORB: a FAST feature detect and extraction algorithm in computer vision. It is said to be 100 times faster than SIFT, 10 times faster than SURF and also reduces count of feature points of result, and therefore well supports real-time calculations. (What's more, it's FREE! Unlike SIFT and SURF, which are protected by patent and could not be used in newest version of OpenCV...)
- Mini-batch K-Means: better than normal K-Means algorithm when faced with large amount of data.
- LSHash: maps similar data into signature vectors in close distances.