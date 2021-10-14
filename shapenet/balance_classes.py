import os
import itertools
from pprint import pprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    DATA_DIR = 'train_data_balanced'
    DATA_DIR = os.path.join(BASE_DIR, DATA_DIR)
    assert os.path.exists(DATA_DIR), f'{DATA_DIR} does not exist!'

    print(f'[I] - Enter {DATA_DIR}')

    N_SAMPLES = 1000

    for SUBDIR in os.listdir(DATA_DIR):
        print(f'\t[I] - Enter {SUBDIR}')
        SUBDIR = os.path.join(DATA_DIR, SUBDIR)
        KEY = lambda x: x.split('.')[-2]
        FILES = [
            list(v) for _, v in itertools.groupby(sorted(os.listdir(SUBDIR), key=KEY), key=KEY)
        ]
        if len(FILES) < N_SAMPLES:
            print(f'\t\t[I] |{SUBDIR.split("/")[-1]}| (={len(FILES)}) < {N_SAMPLES}')
            continue
        for F in FILES[N_SAMPLES:]:
            print(f'\t\t[I] - Remove {F}')
            # os.remove(os.path.join(SUBDIR, F[0]))
            # os.remove(os.path.join(SUBDIR, F[1]))

