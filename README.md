# Image Search Engine as Service

> A simple image search engine based on SIFT, LSHash and K-Means algorithm.
>
> Deployment enabled using Flask.

## Requirements

- Python3
- opencv-python==3.4.2.16
- opencv-contrib-python==3.4.2.16
- lshash3==0.0.8
- numpy==1.17.0
- flask==1.1.1

## Process

### Build index of source images

- Use SIFT to extract feature points and corresponding feature vectors from each image.
- Apply K-Means algorithm, aggregate feature vectors and find K centeroids (key features).
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

## Known problems

- SIFT is a bit slow, processing about 5-10 iterations per second.
- Support for large amount of images (maybe over 200,000) is not considered, as we keep the feature vectors of all images in memory all thee time.