import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')

import json
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def resolve_glove_dir() -> Path:
    env_glove_dir = os.environ.get('GLOVE_DIR')
    if env_glove_dir:
        return Path(env_glove_dir).expanduser()

    candidates = [
        SCRIPT_DIR / 'data' / 'glove',
        REPO_ROOT / 'data' / 'glove',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


GLOVE_DIR = resolve_glove_dir()
GLOVE_DIR.mkdir(parents=True, exist_ok=True)
GLOVE_6B_URL = 'https://nlp.stanford.edu/data/glove.6B.zip'
_GLOVE_CACHE = {}
_DATA_CACHE = {}
_WORD_INDEX = None


def ensure_glove_6b(download_dir: Path) -> Path:
    zip_path = download_dir / 'glove.6B.zip'
    extracted_dir = download_dir / 'glove.6B'

    if extracted_dir.exists() and any(extracted_dir.glob('glove.6B.*d.txt')):
        return extracted_dir

    if not zip_path.exists():
        print(f'Downloading GloVe 6B to: {zip_path}')
        urllib.request.urlretrieve(GLOVE_6B_URL, zip_path)

    print(f'Extracting: {zip_path} -> {download_dir}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(download_dir)

    if extracted_dir.exists():
        return extracted_dir
    if any(download_dir.glob('glove.6B.*d.txt')):
        return download_dir

    raise FileNotFoundError('GloVe files not found after extraction.')


def load_glove_vectors(glove_folder: Path, embedding_dimension: int) -> dict:
    cache_key = (str(glove_folder.resolve()), embedding_dimension)
    if cache_key in _GLOVE_CACHE:
        return _GLOVE_CACHE[cache_key]

    glove_path = glove_folder / f'glove.6B.{embedding_dimension}d.txt'
    if not glove_path.exists():
        raise FileNotFoundError(f'Missing {glove_path}.')
    embeddings_index = {}
    with glove_path.open(encoding='utf8') as f:
        for line in f:
            word, *vec = line.split()
            embeddings_index[word] = np.asarray(vec, dtype='float32')
    _GLOVE_CACHE[cache_key] = embeddings_index
    return embeddings_index


def get_imdb_word_index() -> dict:
    global _WORD_INDEX
    if _WORD_INDEX is None:
        _WORD_INDEX = imdb.get_word_index()
    return _WORD_INDEX


def build_embedding_matrix(training_vocabulary_size: int, embedding_dimension: int) -> np.ndarray:
    glove_folder = ensure_glove_6b(GLOVE_DIR)
    embeddings_index = load_glove_vectors(glove_folder, embedding_dimension)
    word_index = get_imdb_word_index()
    embedding_matrix = np.zeros((training_vocabulary_size, embedding_dimension), dtype='float32')

    for word, raw_idx in word_index.items():
        idx = raw_idx + 3
        if idx >= training_vocabulary_size:
            continue
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[idx] = vec
    return embedding_matrix


def prep_data(training_vocabulary_size: int, max_text_length: int):
    cache_key = (training_vocabulary_size, max_text_length)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    (X_tr, y_tr), (X_te, y_te) = imdb.load_data(num_words=training_vocabulary_size)
    X = np.concatenate([X_tr, X_te], axis=0)
    y = np.concatenate([y_tr, y_te], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )
    X_train = pad_sequences(X_train, maxlen=max_text_length, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_text_length, padding='post', truncating='post')
    _DATA_CACHE[cache_key] = (X_train, X_test, y_train, y_test)
    return _DATA_CACHE[cache_key]


def build_model(max_text_length, training_vocabulary_size, embedding_dimension, embedding_matrix,
                dense_units=None, dropout_rate=0.0, l2=None, trainable=True):
    layers = [
        Input(shape=(max_text_length,), dtype='int32'),
        Embedding(
            input_dim=training_vocabulary_size,
            output_dim=embedding_dimension,
            weights=[embedding_matrix],
            mask_zero=True,
            trainable=trainable,
        ),
        GlobalAveragePooling1D(),
    ]
    if dense_units is not None:
        reg = tf.keras.regularizers.l2(l2) if l2 else None
        layers.append(Dense(dense_units, activation='relu', kernel_regularizer=reg))
        if dropout_rate and dropout_rate > 0:
            layers.append(Dropout(dropout_rate))
    layers.append(Dense(1, activation='sigmoid'))
    return Sequential(layers)


def train_eval(title, model, X_train, y_train, X_test, y_test, lr=1e-3, use_reduce=False,
               epochs=40, batch_size=128, patience=5):
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, min_delta=1e-4, restore_best_weights=True)]
    if use_reduce:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_delta=1e-5, min_lr=1e-8, verbose=0))

    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    hist = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    elapsed = time.time() - start
    min_idx = int(np.argmin(hist.history['val_loss']))
    val_acc = float(hist.history['val_accuracy'][min_idx])
    val_loss = float(hist.history['val_loss'][min_idx])
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    out = {
        'title': title,
        'val_acc_at_min_loss': val_acc,
        'min_val_loss': val_loss,
        'epoch_at_min_loss': min_idx + 1,
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'gap': abs(val_acc - float(test_acc)),
        'elapsed_sec': elapsed,
    }
    print(out)
    return out


def attach_configs(result: dict, dataset_config: dict, model_config: dict, train_config: dict) -> dict:
    result['dataset_config'] = dict(dataset_config)
    result['model_config'] = dict(model_config)
    result['train_config'] = dict(train_config)
    return result


def main():
    all_results = []

    # Problem 1 baseline settings
    baseline_dataset_config = {
        'vocab': 50_000,
        'max_len': 500,
        'emb_dim': 100,
    }
    X_train, X_test, y_train, y_test = prep_data(
        baseline_dataset_config['vocab'],
        baseline_dataset_config['max_len'],
    )
    emb_matrix = build_embedding_matrix(
        baseline_dataset_config['vocab'],
        baseline_dataset_config['emb_dim'],
    )

    p1_runs = []
    p1_specs = [
        {
            'title': 'P1-baseline-unfrozen',
            'model_config': {'dense_units': None, 'dropout_rate': 0.0, 'l2': None, 'trainable': True},
            'train_config': {'lr': 1e-3, 'use_reduce': False},
        },
        {
            'title': 'P1-dense64-drop03-l2-1e4',
            'model_config': {'dense_units': 64, 'dropout_rate': 0.3, 'l2': 1e-4, 'trainable': True},
            'train_config': {'lr': 1e-3, 'use_reduce': False},
        },
        {
            'title': 'P1-dense64-drop02-l2-1e4-lr1e4-reduce',
            'model_config': {'dense_units': 64, 'dropout_rate': 0.2, 'l2': 1e-4, 'trainable': True},
            'train_config': {'lr': 1e-4, 'use_reduce': True},
        },
    ]
    for spec in p1_specs:
        result = train_eval(
            spec['title'],
            build_model(
                baseline_dataset_config['max_len'],
                baseline_dataset_config['vocab'],
                baseline_dataset_config['emb_dim'],
                emb_matrix,
                **spec['model_config'],
            ),
            X_train,
            y_train,
            X_test,
            y_test,
            **spec['train_config'],
        )
        p1_runs.append(
            attach_configs(result, baseline_dataset_config, spec['model_config'], spec['train_config'])
        )

    best_p1 = max(p1_runs, key=lambda x: x['val_acc_at_min_loss'])
    print('BEST_P1', best_p1)
    all_results.extend(p1_runs)

    # Problem 2: vary max len and vocab, keep emb dim 100 and reuse the actual best architecture from P1
    p2_runs = []
    p2_specs = [
        {
            'title': 'P2-maxlen700',
            'dataset_config': {'vocab': 50_000, 'max_len': 700, 'emb_dim': 100},
            'train_config': {'lr': 1e-4, 'use_reduce': True},
        },
        {
            'title': 'P2-vocab70000',
            'dataset_config': {'vocab': 70_000, 'max_len': 500, 'emb_dim': 100},
            'train_config': {'lr': 1e-4, 'use_reduce': True},
        },
        {
            'title': 'P2-vocab70000-maxlen700',
            'dataset_config': {'vocab': 70_000, 'max_len': 700, 'emb_dim': 100},
            'train_config': {'lr': 1e-4, 'use_reduce': True},
        },
    ]
    for spec in p2_specs:
        dataset_config = spec['dataset_config']
        X_train2, X_test2, y_train2, y_test2 = prep_data(
            dataset_config['vocab'],
            dataset_config['max_len'],
        )
        emb_matrix2 = build_embedding_matrix(dataset_config['vocab'], dataset_config['emb_dim'])
        model2 = build_model(
            dataset_config['max_len'],
            dataset_config['vocab'],
            dataset_config['emb_dim'],
            emb_matrix2,
            **best_p1['model_config'],
        )
        result = train_eval(
            spec['title'],
            model2,
            X_train2,
            y_train2,
            X_test2,
            y_test2,
            **spec['train_config'],
        )
        p2_runs.append(
            attach_configs(result, dataset_config, best_p1['model_config'], spec['train_config'])
        )

    best_p2 = max(p2_runs, key=lambda x: x['val_acc_at_min_loss'])
    print('BEST_P2', best_p2)
    all_results.extend(p2_runs)

    # Problem 3: vary embedding dimension while keeping the best data settings from Problem 2
    best_p2_dataset_config = dict(best_p2['dataset_config'])
    X_train3, X_test3, y_train3, y_test3 = prep_data(
        best_p2_dataset_config['vocab'],
        best_p2_dataset_config['max_len'],
    )

    p3_runs = []
    for dim in sorted({50, best_p2_dataset_config['emb_dim'], 300}):
        dataset_config = dict(best_p2_dataset_config)
        dataset_config['emb_dim'] = dim
        emb_matrix3 = build_embedding_matrix(dataset_config['vocab'], dim)
        model3 = build_model(
            dataset_config['max_len'],
            dataset_config['vocab'],
            dim,
            emb_matrix3,
            **best_p2['model_config'],
        )
        train_config = dict(best_p2['train_config'])
        result = train_eval(
            f'P3-emb{dim}',
            model3,
            X_train3,
            y_train3,
            X_test3,
            y_test3,
            **train_config,
        )
        p3_runs.append(
            attach_configs(result, dataset_config, best_p2['model_config'], train_config)
        )

    best_p3 = max(p3_runs, key=lambda x: x['val_acc_at_min_loss'])
    print('BEST_P3', best_p3)
    all_results.extend(p3_runs)

    out_path = SCRIPT_DIR / 'hw7_results.json'
    out_path.write_text(json.dumps({
        'best_p1': best_p1,
        'best_p2': best_p2,
        'best_p3': best_p3,
        'all_results': all_results,
    }, indent=2), encoding='utf8')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
