IDEA: use a combination of convolutional and recurrent networks to
derive a set a features for a song classification task (e.g. predicting
genre/style or mood)

Then use variations of these features along with new songs as input to
alter the style of the song, similar to http://arxiv.org/abs/1508.06576
for artistic images.


Plan is to split a song into 5-10 second chunks (enough to keep a
coherent picture of the timbre and style of the music) and use a deep
convolutional network to generate a 1-dimensional embedding (currently
I have a small version of GoogleNet/Inception written out, but I may need
a lot more data for that).

These embeddings are then sequentially fed into a recurrent neural
network with either LSTM or GRU cells. Output can either be a
concatenation of all the RNN outputs or the last output.

Two initial things to try: inputting song as a waveform and doing 1-D
convolution, or inputting song as a spectrogram. It seems that ideally
with enough data the waveform should work, and the network should be
able to "learn" how to take the FFT. However, this seems implausible
given the amount of data I currently have (~7000 songs). Plus, humans
more or less have the FFT hardwired into our inner ears, and don't use
extensive brainpower to compute it.

Songs currently come from the MIREX competition, but I think I can get a
lot more using the freesound api, and almost all music I get is going to
come prelabeled with some sort of metadata, and even if it's not I can
look up the artist/track on an internet database to retrieve the
metadata.


I've created a time-distributed 1D convolutional layer in Keras in the
file `keras_extra_layers.py`
