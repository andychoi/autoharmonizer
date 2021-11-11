import numpy as np
import math
from music21 import converter
from music21 import note
from music21 import chord
import loader

def get_melody_list(melody_part):
    melody_list = []
    for element in melody_part.recurse():
        if isinstance(element, note.Note):
            melody_list.append(element)
    return melody_list

def get_chord_list(chord_part):
    chord_list = []
    for element in chord_part.recurse():
        if isinstance(element, chord.Chord):
            chord_list.append(element)
    return chord_list

def get_chord_vec_list(chord_list):
    chord_vec_list = []
    for chord in chord_list:
        chord_vec = loader.chord2vec(chord)
        for i in range(len(chord_vec)):
            if i != 0 and chord_vec[i] == 0:
                del chord_vec[i]
                continue
            chord_vec[i] -= 1 # Important
            if i != 0: chord_vec[i] = (chord_vec[i] + chord_vec[0]) % 12
        chord_vec_list.append(chord_vec)
    return chord_vec_list

def tonal_centroid(notes):
    fifths_lookup = {9:[1.0, 0.0], 2:[math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)], 7:[math.cos(2.0 * math.pi / 6.0), math.sin(2.0 * math.pi / 6.0)],
                     0:[0.0, 1.0], 5:[math.cos(4.0 * math.pi / 6.0), math.sin(4.0 * math.pi / 6.0)], 10:[math.cos(5.0 * math.pi / 6.0), math.sin(5.0 * math.pi / 6.0)],
                     3:[-1.0, 0.0], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 1:[math.cos(8.0 * math.pi / 6.0), math.sin(8.0 * math.pi / 6.0)],
                     6:[0.0, -1.0], 11:[math.cos(10.0 * math.pi / 6.0), math.sin(10.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}
    minor_thirds_lookup = {3:[1.0, 0.0], 7:[1.0, 0.0], 11:[1.0, 0.0],
                           0:[0.0, 1.0], 4:[0.0, 1.0], 8:[0.0, 1.0],
                           1:[-1.0, 0.0], 5:[-1.0, 0.0], 9:[-1.0, 0.0],
                           2:[0.0, -1.0], 6:[0.0, -1.0], 10:[0.0, -1.0]}
    major_thirds_lookup = {0:[0.0, 1.0], 3:[0.0, 1.0], 6:[0.0, 1.0], 9:[0.0, 1.0],
                           2:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 5:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 8:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)], 11:[math.cos(7.0 * math.pi / 6.0), math.sin(7.0 * math.pi / 6.0)],
                           1:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 4:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 7:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)], 10:[math.cos(11.0 * math.pi / 6.0), math.sin(11.0 * math.pi / 6.0)]}

    fifths = [0.0, 0.0]
    minor = [0.0, 0.0]
    major = [0.0, 0.0]
    r1 =1
    r2 =1
    r3 = 0.5
    if notes:
        for note in notes:
            for i in range(2):
                fifths[i] += r1 * fifths_lookup[note][i]
                minor[i] += r2 * minor_thirds_lookup[note][i]
                major[i] += r3 * major_thirds_lookup[note][i]
        for i in range(2):
            fifths[i] /= len(notes)
            minor[i] /= len(notes)
            major[i] /= len(notes)

    return fifths + minor + major

def get_CHE_and_CC(chord_list):
    cnt = {}
    for chord in chord_list:
        chord_vec = loader.chord2vec(chord)
        chord_str = ''.join([str(i) for i in chord_vec])
        if chord_str in cnt:
            cnt[chord_str] += 1
        else: cnt[chord_str] = 1
    
    CC = len(cnt)

    CHE = 0
    for key, value in cnt.items():
        value /= len(chord_list)
        CHE += - value * np.log(value+1e-6)
    
    return CHE, CC

def get_CTD(chord_list):
    chord_vec_list = get_chord_vec_list(chord_list)
    score = 0
    for i in range(len(chord_vec_list) - 1):
        score += np.sqrt(np.sum((np.asarray(tonal_centroid(chord_vec_list[i+1])) - np.asarray(tonal_centroid(chord_vec_list[i]))) ** 2))

    if (len(chord_vec_list) - 1) == 0: return 0
    return score / (len(chord_vec_list) - 1)

def get_CTnCTR(melody_list, chord_list):
    chord_vec_list = get_chord_vec_list(chord_list)
    note_index = 0
    c = 0
    p = 0
    n = 0
    for chord_index in range(len(chord_list)):
        if (note_index >= len(melody_list)): break
        while ((melody_list[note_index].offset < chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
            and (melody_list[note_index].offset >= chord_list[chord_index].offset)):
            if melody_list[note_index].pitch.pitchClass in chord_vec_list[chord_index]:
                c += melody_list[note_index].quarterLength
            else:
                n += melody_list[note_index].quarterLength
                j = 1 # j'th note after note_index
                if (note_index + j >= len(melody_list)): break
                while ((melody_list[note_index + j].offset < chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
                    and (melody_list[note_index + j].offset >= chord_list[chord_index].offset)):
                    if melody_list[note_index + j].pitch.pitchClass != melody_list[note_index].pitch.pitchClass:
                        if ((melody_list[note_index + j].pitch.pitchClass in chord_vec_list[chord_index])
                            and (abs(melody_list[note_index + j].pitch.pitchClass - melody_list[note_index].pitch.pitchClass <= 2))):
                            p += melody_list[note_index].quarterLength
                            break
                    j += 1
                    if (note_index + j >= len(melody_list)): break
            note_index += 1
            if (note_index >= len(melody_list)): break
    
    if c + n == 0: return 0
    return (c+p) / (c+n)

def get_PCS(melody_list, chord_list):
    chord_vec_list = get_chord_vec_list(chord_list)
    note_index = 0
    score = 0
    cnt = 0
    for chord_index in range(len(chord_list)):
        if (note_index >= len(melody_list)): break
        while ((melody_list[note_index].offset < chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
            and (melody_list[note_index].offset >= chord_list[chord_index].offset)):
            m = melody_list[note_index].pitch.pitchClass
            dur = melody_list[note_index].quarterLength
            for c in chord_vec_list[chord_index]:
                if abs(m - c) == 0 or abs(m - c) == 3 or abs(m - c) == 4 or abs(m - c) == 7 or abs(m - c) == 8 or abs(m - c) == 9 or abs(m - c) == 5:
                    if abs(m - c) == 5:
                        cnt += dur
                    else:
                        cnt += dur
                        score += dur
                else:
                    cnt += dur
                    score += -dur
            note_index += 1
            if (note_index >= len(melody_list)): break

    if cnt == 0: return 0
    return score / cnt

def get_MCTD(melody_list, chord_list):
    chord_vec_list = get_chord_vec_list(chord_list)
    note_index = 0
    score = 0
    cnt = 0
    for chord_index in range(len(chord_list)):
        if (note_index >= len(melody_list)): break
        while ((melody_list[note_index].offset < chord_list[chord_index].offset + chord_list[chord_index].quarterLength)
            and (melody_list[note_index].offset >= chord_list[chord_index].offset)):
            m = melody_list[note_index].pitch.pitchClass
            dur = melody_list[note_index].quarterLength
            
            score += np.sqrt(np.sum((np.asarray(tonal_centroid([m])) - np.asarray(tonal_centroid(chord_vec_list[chord_index])))) ** 2) * dur
            cnt += dur
            
            note_index += 1
            if (note_index >= len(melody_list)): break

    if cnt == 0: return 0
    return score / cnt

if __name__ == "__main__":

    filename = r'outputs/test.mid'
    score = converter.parse(filename)
    melody_part = score.parts[0].flat
    chord_part = score.parts[1].flat

    melody_list = get_melody_list(melody_part)
    chord_list = get_chord_list(chord_part)

    CHE, CC = get_CHE_and_CC(chord_list)
    CTD = get_CTD(chord_list)
    CTnCTR = get_CTnCTR(melody_list, chord_list)
    PCS = get_PCS(melody_list, chord_list)
    MCTD = get_MCTD(melody_list, chord_list)

    print('CHE = ', CHE)
    print('CC = ', CC)
    print('CTD = ', CTD)
    print('CTnCTR = ', CTnCTR)
    print('PCS = ', PCS)
    print('MCTD = ', MCTD)

