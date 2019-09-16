import os
import json
import logging
from flask import Flask, request
from argparse import ArgumentParser
from engine import ImageSearchEngine

app = Flask(__name__)
engine = ImageSearchEngine()
logger = logging.getLogger('Image search:service')
logger.setLevel(logging.INFO)
img_types = {'.jpg', '.jpeg', '.bmp', '.png', '.gif', '.webp'}
parser = ArgumentParser('Deploy image search engine as a local service.')
parser.add_argument('--host', type=str, default='localhost',
                    help='host of your image search service')
parser.add_argument('--port', '-p', type=int, default=7777,
                    help='port of your image search service')
parser.add_argument('--data', '-d', type=str, required=True,
                    help='image folder which contains source images')
parser.add_argument('--k', '-k', type=int, required=True,
                    help='number of key features used in K-Means')
parser.add_argument('--hash_size', type=int, default=10,
                    help='length of resulting binary hash array')
parser.add_argument('--num_hashtables', type=int, required=False,
                    help='number of hashtables for multiple lookups.')
parser.add_argument('--store_file', '-s', type=str, required=False,
                    help='the path to the .npz file random matrices are stored')
parser.add_argument('--overwrite', '-o', type=bool, required=False,
                    help='whether to overwrite the matrices file if exist')


def get_img_list(data_dir: str) -> list:
    img_list = []
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if name[name.rfind('.'):] in img_types:
                img_list.append(os.path.join(root, name))
    return img_list


@app.route('/imgsearch', methods=['POST'])
def search() -> str:
    res = []
    try:
        img_name = request.form.get('image')
        num_results = request.form.get('n', None)
        dist_func = request.form.get('distance', None)
        res = engine.search(img_name, num_results, dist_func)
        logger.info('Search picture: {}, find {} results.'.format(img_name,
                                                                  len(res)))
    except Exception as e:
        logger.warning(e)
    return json.dumps(res)


if __name__ == '__main__':
    args = parser.parse_args()
    images = get_img_list(args.data)
    engine.load_images(images)
    engine.build_index(args.k, args.hash_size, args.num_hashtables,
                       args.store_file, args.overwrite)
    engine.dump()
    app.run(host=args.host, port=args.port)
