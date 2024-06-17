import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from datasets import load_from_disk
from pathlib import Path


def process_shard(shard_path, ngram_size):
    print(shard_path)
    shard_dataset = load_from_disk(shard_path)
    shard_counts = Counter()
    shard_prefix_counts = Counter()
    for idx, seq in enumerate(shard_dataset):
        if idx % 100000 == 0:
            print(f"{idx}/{len(shard_dataset)}: {shard_path}")
        ids = seq["input_ids"]
        for i in range(len(ids) - ngram_size + 1):
            ngram = tuple(ids[i : i + ngram_size])
            shard_counts[ngram] += 1
            prefix = ngram[:-1]
            shard_prefix_counts[prefix] += 1
    return shard_counts, shard_prefix_counts


def process_domain(domain_path, ngram_size=2):
    domain_path = Path(domain_path)
    shard_paths = [shard_path for shard_path in domain_path.iterdir() if shard_path.is_dir()]
    if len(shard_paths) == 0:
        shard_paths = [domain_path]
    ngram_counts = Counter()
    prefix_counts = Counter()

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_shard,
                shard_path,
                ngram_size,
            )
            for shard_path in shard_paths
        ]
        for future in futures:
            shard_ngram_counts, shard_prefix_counts = future.result()
            ngram_counts.update(shard_ngram_counts)
            prefix_counts.update(shard_prefix_counts)

    total_entropy = 0.0
    total_ngrams = sum(ngram_counts.values())
    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        ngram_prob = count / total_ngrams
        cond_prob = count / prefix_counts[prefix]
        total_entropy += ngram_prob * (
            -np.log(cond_prob)
        )

    ngram_counts_file = os.path.join(domain_path, f"{ngram_size}gram.pkl")
    with open(ngram_counts_file, "wb") as f:
        pickle.dump(ngram_counts, f)

    prefix_counts_file = os.path.join(
        domain_path, f"{ngram_size}gram_prefix.pkl"
    )
    with open(prefix_counts_file, "wb") as f:
        pickle.dump(prefix_counts, f)

    total_entropy_file = os.path.join(
        domain_path, f"{ngram_size}gram_cond_ent.pkl"
    )
    with open(total_entropy_file, "wb") as f:
        pickle.dump(total_entropy, f)

    return total_entropy


if __name__ == "__main__":
    domain_path = sys.argv[1]
    ngram_size = int(sys.argv[2])
    process_domain(domain_path, ngram_size=ngram_size)
