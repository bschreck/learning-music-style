from __future__ import printfunction
from glob import glob
import os
import random
from pydub import AudioSegment
import cStringIO
import data_utils.parse_files as parse

def shuffled_mp3_files(dataset_directory):
    files = [f for f in glob(os.path.join(dataset_directory,"*.mp3"))]
    random.shuffle(files)
    return files
def batched_audio(dataset_directory,batch_size, audio_clips_per_file):
    files = shuffled_mp3_files(dataset_directory)
    num_files_per_batch = batch_size // audio_clips_per_file
    num_batches_per_epoch = len(files) // (num_files_per_batch)
    for i in xrange(num_batches_per_epoch):
        file_batch = files[i*num_files_per_batch: (i+1)*num_files_per_batch]
        songs = [AudioSegment.from_mp3(mp3_file) for mp3_file in file_batch]
        batch = np.zeros((batch_size
        for song in songs:

wav_str = cStringIO.StringIO()
songs[0].export(wav_str, format="wav")
wav_str.seek(0)

data, bitrate = parse.read_wav_as_np(wav_str)
print data[1000:1200,0]
