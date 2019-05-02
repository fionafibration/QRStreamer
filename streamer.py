import fountaincoding
import argparse
import sys
import qrcode
import time
import os
import re
import io
import base64
from math import ceil, floor
from qrcode.constants import *


def main():
    parser = argparse.ArgumentParser(description='Split a file into multiple streamed QR codes using  '
                                                 'fountain coding scheme')

    parser.add_argument('-b', '--block-size', type=int, default=512)
    parser.add_argument('-d', '--duration', type=int, default=300,
                        help='The duration of each individual frame in the QR gif. '
                             'Default is 300 ms')
    parser.add_argument('-e', '--extra', type=int, default=10,
                        help='The number of extra QR codes to generate '
                             'Default is 10.')
    parser.add_argument('-r', '--error', type=int, choices=[0, 1, 2, 3], default=1,
                        help='The level of error correction. Default is 1')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'),
                        default=sys.stdin.buffer, help='The filename or path of file to stream. '
                                                       'Default is stdin.')

    args = parser.parse_args()

    input_data = args.infile.read()

    running_extra = int(args.extra)

    print('Encoding data...')

    while True:
        try:
            data, score, compressed = fountaincoding.optimal_encoding(io.BytesIO(input_data), args.block_size,
                                                                  extra=floor(running_extra))
            break
        except Exception:
            running_extra += 1
            print('Increasing extra QR codes so decoding is possible...')

    current_path = os.getcwd()

    timestamp = str(round(time.time()))

    os.mkdir(os.path.join(current_path, timestamp))

    images = []

    print('Generating QR codes...')

    for i, block in enumerate(data):
        qr = qrcode.QRCode(version=None,
                           error_correction={
                               0: ERROR_CORRECT_L,
                               1: ERROR_CORRECT_M,
                               2: ERROR_CORRECT_Q,
                               3: ERROR_CORRECT_H,
                           }.get(args.error, 1),
                           box_size=8,
                           border=10)
        qr.add_data(base64.b64encode(block))
        qr.make(fit=True)

        img = qr.make_image(fill_color='black', back_color='white')
        images.append(img)
        img.save(os.path.join(current_path, timestamp, '%s_%s.png' % (i, timestamp)))

    with open(os.path.join(current_path, timestamp, re.sub(r'[^\w.]+', '', args.infile.name)), 'wb') as f:
        f.write(input_data)
    args.infile.close()

    images = [j.convert('RGBA') for j in images]

    images[0].save(os.path.join(current_path, timestamp, re.sub(r'[^\w.]+', '', args.infile.name) + '.gif'), format='GIF', save_all=True, append_images=images[1:], duration=args.duration, loop=0)

    optimal_blocks = ceil(len(input_data) / args.block_size)

    print('\nResults:\n-------------------------------')
    print('Generated %s images.' % len(images))
    print('Compressed data' if compressed else 'Did not compress data')
    print('Recommended minimum number of images to print/display/use: %0.0f\n(This is for redundancy)' % ceil(score + 5))
    print('Average number of images taken to decode in testing: %0.2f' % score)
    print('Overhead from encoding was: %0.2f%%' % ((score / optimal_blocks - 1) * 100))


if __name__ == '__main__':
    main()
