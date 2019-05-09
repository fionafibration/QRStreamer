"""
Implementation of a Luby Transform encoder.

This is a type of fountain code, which deals with lossy channels by
sending an infinite stream of statistically correllated packets generated
from a set of blocks into which the source data is divided. In this way,
epensive retransmissions are unecessary, as the receiver will be able
to reconstruct the file with high probability after receiving only
slightly more blocks than one would have to transmit sending the raw
blocks over a lossless channel.

See
D.J.C, MacKay, 'Information theory, inference, and learning algorithms'.
Cambridge University Press, 2003
for reference.

MIT License

Copyright (c) [2015] [Anson Rosenthal]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import io
import zlib
from random import randint
from struct import pack, unpack
from math import log, floor, sqrt, ceil
from collections import defaultdict
from random import choice

DEFAULT_C = 0.1
DEFAULT_DELTA = 0.5

# Parameters for Pseudorandom Number Generator
PRNG_A = 16807
PRNG_M = (1 << 31) - 1
PRNG_MAX_RAND = PRNG_M - 1


def gen_tau(s, k, delta):
    """The Robust part of the RSD, we precompute an
    array for speed
    """
    pivot = floor(k / s)
    return [s / k * 1 / d for d in range(1, pivot)] \
           + [s / k * log(s / delta)] \
           + [0 for d in range(pivot, k)]


def gen_rho(k):
    """The Ideal Soliton Distribution, we precompute
    an array for speed
    """
    return [1 / k] + [1 / (d * (d - 1)) for d in range(2, k + 1)]


def gen_mu(k, delta, c):
    """The Robust Soliton Distribution on the degree of
    transmitted blocks
    """

    S = c * log(k / delta) * sqrt(k)
    tau = gen_tau(S, k, delta)
    rho = gen_rho(k)
    normalizer = sum(rho) + sum(tau)
    return [(rho[d] + tau[d]) / normalizer for d in range(k)]


def gen_rsd_cdf(k, delta, c):
    """The CDF of the RSD on block degree, precomputed for
    sampling speed"""

    mu = gen_mu(k, delta, c)
    return [sum(mu[:d + 1]) for d in range(k)]


class PRNG(object):
    """A Pseudorandom Number Generator that yields samples
    from the set of source blocks using the RSD degree
    distribution described above.
    """

    def __init__(self, params):
        """Provide RSD parameters on construction
        """

        self.state = None  # Seed is set by interfacing code using set_seed
        K, delta, c = params
        self.K = K
        self.cdf = gen_rsd_cdf(K, delta, c)

    def _get_next(self):
        """Executes the next iteration of the PRNG
        evolution process, and returns the result
        """

        self.state = PRNG_A * self.state % PRNG_M
        return self.state

    def _sample_d(self):
        """Samples degree given the precomputed
        distributions above and the linear PRNG output
        """

        p = self._get_next() / PRNG_MAX_RAND
        for ix, v in enumerate(self.cdf):
            if v > p:
                return ix + 1
        return ix + 1

    def set_seed(self, seed):
        """Reset the state of the PRNG to the
        given seed
        """

        self.state = seed

    def get_src_blocks(self, seed=None):
        """Returns the indices of a set of `d` source blocks
        sampled from indices i = 1, ..., K-1 uniformly, where
        `d` is sampled from the RSD described above.
        """

        if seed:
            self.state = seed

        blockseed = self.state
        d = self._sample_d()
        have = 0
        nums = set()
        while have < d:
            num = self._get_next() % self.K
            if num not in nums:
                nums.add(num)
                have += 1

        return blockseed, d, nums


def _split_file(data, blocksize):
    """Block file byte contents into blocksize chunks, padding last one if necessary
    """

    blocks = [int.from_bytes(data[i:i+blocksize].ljust(blocksize, b'0'), sys.byteorder)
            for i in range(0, len(data), blocksize)]
    return len(data), blocks


def encoder(f, blocksize, magic_byte, seed=None, c=DEFAULT_C, delta=DEFAULT_DELTA):
    """Generates an infinite sequence of blocks to transmit
    to the receiver
    """

    # Generate seed if not provided
    if seed is None:
        seed = randint(0, 1 << 31 - 1)

    # get file blocks
    filesize, blocks = _split_file(f.read(), blocksize)

    # init stream vars
    K = len(blocks)
    prng = PRNG(params=(K, delta, c))
    prng.set_seed(seed)

    # block generation loop
    while True:
        blockseed, d, ix_samples = prng.get_src_blocks()
        block_data = 0
        for ix in ix_samples:
            block_data ^= blocks[ix]

        # Generate blocks of XORed data in network byte order
        block = (magic_byte, filesize, blocksize, blockseed, int.to_bytes(block_data, blocksize, sys.byteorder))
        yield pack('!BIII%ss' % blocksize, *block)


# Check node in graph
class CheckNode(object):
    def __init__(self, src_nodes, check):

        self.check = check
        self.src_nodes = src_nodes

    def __repr__(self):
        return "CheckNode(%s)" % self.src_nodes

class BlockGraph(object):
    """Graph on which we run Belief Propagation to resolve
    source node data
    """

    def __init__(self, num_blocks):
        self.checks = defaultdict(list)
        self.num_blocks = num_blocks
        self.eliminated = {}

    def add_block(self, nodes, data):
        """Adds a new check node and edges between that node and all
        source nodes it connects, resolving all message passes that
        become possible as a result.
        """

        # We can eliminate this source node
        if len(nodes) == 1:
            to_eliminate = list(self.eliminate(next(iter(nodes)), data))

            # Recursively eliminate all nodes that can now be resolved
            while len(to_eliminate):
                other, check = to_eliminate.pop()
                to_eliminate.extend(self.eliminate(other, check))
        else:

            # Pass messages from already-resolved source nodes
            for node in list(nodes):
                if node in self.eliminated:
                    nodes.remove(node)
                    data ^= self.eliminated[node]

            # Resolve if we are left with a single non-resolved source node
            if len(nodes) == 1:
                return self.add_block(nodes, data)
            else:

                # Add edges for all remaining nodes to this check
                check = CheckNode(nodes, data)
                for node in nodes:
                    self.checks[node].append(check)

        # Are we done yet?
        return len(self.eliminated) >= self.num_blocks

    def eliminate(self, node, data):
        """Resolves a source node, passing the message to all associated checks
        """

        # Cache resolved value
        self.eliminated[node] = data
        others = self.checks[node]

        del self.checks[node]

        # Pass messages to all associated checks
        for check in others:
            check.check ^= data
            check.src_nodes.remove(node)

            # Yield all nodes that can now be resolved
            if len(check.src_nodes) == 1:
                yield (next(iter(check.src_nodes)), check.check)


class LTDecoder(object):
    def __init__(self, c=DEFAULT_C, delta=DEFAULT_DELTA):
        self.c = c
        self.delta = delta
        self.K = 0
        self.filesize = 0
        self.blocksize = 0
        self.done = False

        self.compressed = False

        self.block_graph = None
        self.prng = None
        self.initialized = False

    def is_done(self):
        return self.done

    def consume_block(self, lt_block):
        (magic_byte, filesize, blocksize, blockseed), block = lt_block

        if magic_byte & 0x01:
            self.compressed = True

        # first time around, init things
        if not self.initialized:
            self.filesize = filesize
            self.blocksize = blocksize

            self.K = ceil(filesize / blocksize)
            self.block_graph = BlockGraph(self.K)
            self.prng = PRNG(params=(self.K, self.delta, self.c))
            self.initialized = True

        # Run PRNG with given seed to figure out which blocks were XORed to make received data
        _, _, src_blocks = self.prng.get_src_blocks(seed=blockseed)

        # If BP is done, stop
        self.done = self._handle_block(src_blocks, block)
        return self.done, self.compressed

    def decode_bytes(self, block_bytes):
        header = unpack('!BIII', block_bytes[:13])
        data = int.from_bytes(block_bytes[13:], 'big')
        return self.consume_block((header, data))

    def bytes_dump(self):
        buffer = io.BytesIO()
        self._stream_dump(buffer)
        raw_data = buffer.getvalue()
        if self.compressed:
            return zlib.decompress(raw_data)
        else:
            return raw_data

    def _stream_dump(self, out_stream):

        # Iterate through blocks, stopping before padding junk
        for ix, block_bytes in enumerate(map(lambda p: int.to_bytes(p[1], self.blocksize, 'big'),
                                             sorted(self.block_graph.eliminated.items(), key=lambda p: p[0]))):
            if ix < self.K - 1 or self.filesize % self.blocksize == 0:
                out_stream.write(block_bytes)
            else:
                out_stream.write(block_bytes[:self.filesize % self.blocksize])

    def _handle_block(self, src_blocks, block):
        """What to do with new block: add check and pass
        messages in graph
        """
        return self.block_graph.add_block(src_blocks, block)


def encode_and_compress(f, block_size, extra=0, compression=None, **kwargs):
    """
    Tests decoding 64 times to get an idea of how many blocks it takes to decode
    Also possibly compress data to be decoded.

    :param f: File object of data to be read off of
    :param block_size: Size of individual blocks
    :param extra: Extra blocks to generate (usually choose 5-10)
    :param compression: None for automatic compression, True or False to manually enable it.
    :param kwargs: Extra arguments for the encoder.
    :return: A list of blocks, a float scoring the decoder efficiency, and a bool of whether the data was compressed
    """

    input_data = f.read()

    magic_byte = 0x00

    compressed = zlib.compress(input_data, level=9)

    if compression is True:
        processed = compressed
        magic_byte ^= 0x01
    elif compression is False:
        processed = input_data
    elif len(compressed) < len(input_data):
        processed = compressed
        magic_byte ^= 0x01
    else:
        processed = input_data

    if len(processed) // block_size < 15:
        extra += 6

    enc = encoder(io.BytesIO(processed), block_size, magic_byte, **kwargs)

    encoded_test_data = [enc.__next__() for _ in range(ceil(len(processed) / block_size) + extra)]
    
    times_to_finish = []
    
    for i in range(64):
        dec = LTDecoder()

        num_blocks_fed = 0

        possible_blocks = list(range(len(encoded_test_data)))

        while True:
            block_pos = choice(possible_blocks)
            possible_blocks.remove(block_pos)
            dec.decode_bytes(encoded_test_data[block_pos])
            if dec.is_done():
                break
            num_blocks_fed += 1
        times_to_finish.append(num_blocks_fed)

    return encoded_test_data, sum(times_to_finish) / len(times_to_finish), magic_byte & 0x01 > 0, processed


if __name__ == '__main__':
    block_size = 512
    input_data = bytes([randint(0, 255) for _ in range(40000)])

    data, score, compressed, _ = encode_and_compress(io.BytesIO(input_data), block_size, len(input_data))

    print('Optimized data encoding!')

    times_to_finish = []

    for i in range(512):
        dec = LTDecoder()

        num_blocks_fed = 0

        possible_blocks = list(range(len(data)))

        while True:
            block_pos = choice(possible_blocks)
            possible_blocks.remove(block_pos)
            dec.decode_bytes(data[block_pos])
            if dec.is_done():
                break
            num_blocks_fed += 1
        times_to_finish.append(num_blocks_fed)
        assert dec.bytes_dump() == input_data

    optimal_blocks = ceil(len(input_data) / block_size)
    average_blocks = sum(times_to_finish) / len(times_to_finish)
    max_blocks = max(times_to_finish)
    min_blocks = min(times_to_finish)

    print('Summary:\n--------------------------------')

    print('Total blocks: %s' % len(data))
    print('Compressed data' if compressed else 'Uncompressed data')
    print('Optimization function score: %s' % score)
    print('Max blocks: %s' % max_blocks)
    print('Avg. blocks: %s' % average_blocks)
    print('Min blocks: %s' % min_blocks)
    print('Optimal blocks: %s' % optimal_blocks)
    print('LT coding overhead: %0.2f%%' % ((average_blocks / optimal_blocks - 1) * 100))
    print('Done testing!')

