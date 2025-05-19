import os
import warnings
import pickle
import numpy as np
from config import *
from music21 import *
from tqdm import trange
from copy import deepcopy
from model import build_model
from samplings import gamma_sampling
from loader import get_filenames, convert_files
from tensorflow.python.keras.utils.np_utils import to_categorical

# force CPU-only and suppress TF warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# Load chord types once
with open(CHORD_TYPES_PATH, "rb") as fp:
    chord_types = pickle.load(fp)

# Inference batch size
INFER_BATCH_SIZE = 32

def generate_chord(chord_model, melody_data, beat_data, key_data,
                   segment_length=SEGMENT_LENGTH, rhythm_gamma=RHYTHM_DENSITY,
                   chord_per_bar=CHORD_PER_BAR, batch_size=INFER_BATCH_SIZE):

    chord_data = []

    for song_idx, song_melody in enumerate(melody_data):
        padded_melody = segment_length*[0] + song_melody + segment_length*[0]
        padded_beat   = segment_length*[0] + beat_data[song_idx]   + segment_length*[0]
        padded_key    = segment_length*[0] + key_data[song_idx]    + segment_length*[0]
        song_chord    = segment_length * [0]

        n_steps = len(padded_melody) - 2*segment_length

        buf_ml, buf_mr = [], []
        buf_cl, buf_cr = [], []
        buf_chl, buf_info = [], []

        for t in trange(segment_length, len(padded_melody)-segment_length,
                        desc=f"Song {song_idx+1} [{n_steps} steps]"):

            left = slice(t-segment_length, t)
            right = slice(t, t+segment_length)

            ml = to_categorical(padded_melody[left], num_classes=128)[None]
            mr = to_categorical(padded_melody[right][::-1], num_classes=128)[None]

            bl = to_categorical(padded_beat[left], num_classes=5)
            br = to_categorical(padded_beat[right][::-1], num_classes=5)
            kl = to_categorical(padded_key[left], num_classes=16)
            kr = to_categorical(padded_key[right][::-1], num_classes=16)
            cl = np.concatenate((bl, kl), axis=-1)[None]
            cr = np.concatenate((br, kr), axis=-1)[None]

            chl = to_categorical(song_chord[-segment_length:], num_classes=len(chord_types))[None]

            buf_ml.append(ml)
            buf_mr.append(mr)
            buf_cl.append(cl)
            buf_cr.append(cr)
            buf_chl.append(chl)
            buf_info.append((song_chord[-1], padded_beat[t]))

            if len(buf_ml) == batch_size or t == len(padded_melody)-segment_length-1:
                X = [
                    np.vstack(buf_ml),
                    np.vstack(buf_mr),
                    np.vstack(buf_cl),
                    np.vstack(buf_cr),
                    np.vstack(buf_chl)
                ]
                preds = chord_model.predict(X, verbose=0)
                for p, (prev_chord, beat_val) in zip(preds, buf_info):
                    if chord_per_bar:
                        gamma = 1 if beat_val == 4 and prev_chord != song_chord[-1] else 0
                    else:
                        gamma = rhythm_gamma
                    tuned = gamma_sampling(p, [[prev_chord]], [gamma], return_probs=True)
                    cho = np.argmax(tuned, axis=-1)
                    song_chord.append(cho)

                buf_ml.clear(); buf_mr.clear()
                buf_cl.clear(); buf_cr.clear()
                buf_chl.clear(); buf_info.clear()

        chord_data.append(song_chord[segment_length:])

    return chord_data

def watermark(score, filename, water_mark=WATER_MARK):
    if water_mark:
        score.metadata = metadata.Metadata()
        score.metadata.title = filename
        score.metadata.composer = 'harmonized by AutoHarmonizer'
    return score

def export_music(score, beat_data, chord_data, filename,
                 repeat_chord=REPEAT_CHORD, outputs_path=OUTPUTS_PATH,
                 water_mark=WATER_MARK):

    harmony_list = []
    offset = 0.0
    base = os.path.basename(filename)
    stem = '.'.join(base.split('.')[:-1])

    for idx, song_ch in enumerate(chord_data):
        labels = [chord_types[int(c)].replace('N.C.', 'R').replace('bpedal', '-pedal') for c in song_ch]
        pre = None
        for t, lbl in enumerate(labels):
            if lbl != 'R' and (lbl != pre or (repeat_chord and beat_data[idx][t] == 4)):
                cs = harmony.ChordSymbol(lbl)
                cs.offset = offset
                harmony_list.append(cs)
            offset += 0.25
            pre = lbl

    new_measures = []
    offsets = []
    h_idx = 0
    for m in score:
        if isinstance(m, stream.Measure):
            new_m = deepcopy(m)
            offsets.append(m.offset)
            elems = []
            for el in new_m:
                while h_idx < len(harmony_list) and el.offset + m.offset >= harmony_list[h_idx].offset:
                    harmony_list[h_idx].offset -= m.offset
                    elems.append(harmony_list[h_idx])
                    h_idx += 1
                elems.append(el)
            new_m.elements = elems
            new_measures.append(new_m)

    final_score = stream.Score(new_measures)
    for i, m in enumerate(final_score):
        m.offset = offsets[i]

    if water_mark:
        final_score = watermark(final_score, stem)

    final_score.write('mxl', fp=f"{outputs_path}/{stem}.mxl")

if __name__ == "__main__":
    files = get_filenames(input_dir=INPUTS_PATH)
    data = convert_files(files, fromDataset=False)

    model = build_model(SEGMENT_LENGTH, RNN_SIZE, NUM_LAYERS, DROPOUT,
                        WEIGHTS_PATH, training=False)

    for md, bd, kd, score_obj, fname in data:
        chords = generate_chord(model, md, bd, kd)
        export_music(score_obj, bd, chords, fname)
