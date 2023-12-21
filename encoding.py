from mido import MidiFile, MidiTrack, Message, merge_tracks
import os
import json
import copy
import shutil
import pickle



def encode_midi_with_note_events(midi_file_path):
    midi = MidiFile(midi_file_path)
    encoded_events = []

    last_velocity = None
    first_note = False
    granularity = 10

    for msg in merge_tracks(midi.tracks):

        time_left = msg.time

        while time_left > granularity and first_note:

            if time_left >= 1000:
                encoded_events.append('TIME_SHIFT<1000>')
                time_left -= 1000
            else:
                rounded_time = (time_left // granularity) * granularity
                encoded_events.append(f'TIME_SHIFT<{rounded_time}>')
                time_left -= rounded_time


        if msg.type == 'note_on' and msg.velocity > 0:
            first_note = True
            if last_velocity != msg.velocity:
                encoded_events.append(f'SET_VELOCITY<{msg.velocity}>')
                last_velocity = msg.velocity

            encoded_events.append(f'NOTE_ON<{msg.note}>')

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            encoded_events.append(f'NOTE_OFF<{msg.note}>')

    return encoded_events


def decode_events_to_midi(encoded_events, output_midi_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    current_velocity = 64
    current_time = 0

    for event in encoded_events:
        if 'SET_VELOCITY' in event:
            current_velocity = int(event.split('<')[1].split('>')[0])

        elif 'TIME_SHIFT' in event:
            shift_amount = int(event.split('<')[1].split('>')[0])
            current_time += shift_amount

        elif 'NOTE_ON' in event:
            note_num = int(event.split('<')[1].split('>')[0])
            track.append(Message('note_on', note=note_num, velocity=current_velocity, time=current_time))
            current_time = 0

        elif 'NOTE_OFF' in event:
            note_num = int(event.split('<')[1].split('>')[0])
            track.append(Message('note_off', note=note_num, velocity=0, time=current_time))
            current_time = 0

    midi.save(output_midi_path)


def split_into_groups(lst, size):
    if len(lst) < size:
        return [lst]
    else:
        chunks = [lst[i:i + size] for i in range(0, len(lst) - size, size)]
        chunks.append(lst[-size:])
        return chunks


def id_to_event(ids):
    dict = json.load(open('config.json'))
    id2token = dict['id2token']
    tokens = []
    for id in ids:
        tokens.append(id2token[str(id)])
    return tokens


def create_dataset(input_dir, output_json, output_dir):
    dataset = {
        "tokens": [],
    }
    vocab = set()
    for num, filename in enumerate(os.listdir(input_dir)):
        print(num)
        if filename.endswith(".midi") or filename.endswith(".mid"):

            filepath = os.path.join(input_dir, filename)
            tokens = encode_midi_with_note_events(filepath)
            token_set = {token for token in tokens}
            vocab = vocab.union(token_set)

            dataset["tokens"].extend(split_into_groups(tokens, 1000))


    token2id = {token: i for i, token in enumerate(vocab)}
    id2token = {i: token for token, i in token2id.items()}

    dataset["encodings"] = copy.deepcopy(dataset["tokens"])
    for idx, tokens in enumerate(dataset["tokens"]):
        dataset["encodings"][idx] = [token2id[token] for token in tokens]

    config = {
        "token2id": token2id,
        "id2token": id2token
    }

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with open(os.path.join(output_dir, output_json), 'w+') as f:
        json.dump(config, f)

    length = len(dataset['encodings'])
    train_length = int(length * 0.8)
    test_length = int(length * 0.1)
    val_length = int(length * 0.1)
    train = dataset['encodings'][0:train_length]
    test = dataset['encodings'][train_length:train_length + test_length]
    val = dataset['encodings'][train_length + val_length:]

    print(len(train))
    print(len(test))
    print(len(val))

    train_dir = output_dir + '/train'
    test_dir = output_dir + '/test'
    val_dir = output_dir + '/val'
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(val_dir)

    for i, f in enumerate(train):
        file = open(train_dir + '/train-' + str(i) + '.midi.pickle', 'wb')
        pickle.dump(f, file)
        file.close()

    for i, f in enumerate(test):
        file = open(test_dir + '/test-' + str(i) + '.midi.pickle', 'wb')
        pickle.dump(f, file)
        file.close()

    for i, f in enumerate(val):
        file = open(val_dir + '/val-' + str(i) + '.midi.pickle', 'wb')
        pickle.dump(f, file)
        file.close()


if __name__ == '__main__':

    # encoded = encode_midi('/Users/liampilarski/Desktop/MusicGPT/muse.mid')
    # print(len(encoded))
    # encoding_set = {s for s in encoded}
    # # print(encoding_set)
    # # print(len(encoding_set))
    # decode_midi(encoded, 'primer_restored.mid')
    # # decode_events_to_midi([encoded[:250], encoded[250:500]], 'output_midi_file_restored.mid')

    create_dataset("output_jazz", 'config.json', 'dataset_jazz')

