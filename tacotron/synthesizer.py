import os
import wave
from datetime import datetime
import numpy as np
import pyaudio
import sounddevice as sd
import tensorflow as tf
from datasets import audio
from infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence


class Synthesizer:
	def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
		log('Constructing model: %s' % model_name)
		inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
		input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
		with tf.variable_scope('model') as scope:
			self.model = create_model(model_name, hparams)
			self.model.initialize(inputs, input_lengths)
			self.alignments = self.model.alignments
			self.lf0_outputs = self.model.lf0_outputs
			self.mgc_outputs = self.model.mgc_outputs
			self.bap_outputs = self.model.bap_outputs

		self.gta = gta
		self._hparams = hparams
		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0

		log('Loading checkpoint: %s' % checkpoint_path)
		#Memory allocation on the GPU as needed
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		self.session = tf.Session(config=config)
		self.session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, basenames, out_dir, log_dir):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}

		lf0s, mgcs, baps, alignments = self.session.run([self.lf0_outputs, self.mgc_outputs, self.bap_outputs, self.alignments], feed_dict=feed_dict)

		for i, _ in enumerate(lf0s):
			# Write the predicted features to disk
			# Note: outputs files and target ones have same names, just different folders
			np.save(os.path.join(out_dir, 'lf0-{:03d}.npy'.format(basenames[i])), lf0s[i], allow_pickle=False)
			np.save(os.path.join(out_dir, 'mgc-{:03d}.npy'.format(basenames[i])), mgcs[i], allow_pickle=False)
			np.save(os.path.join(out_dir, 'bap-{:03d}.npy'.format(basenames[i])), baps[i], allow_pickle=False)

			if log_dir is not None:
				#save alignments
				plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{:03d}.png'.format(basenames[i])),
					info='{}'.format(texts[i]), split_title=True)

				#save wav
				wav = audio.synthesize(lf0[i], mgcs[i], baps[i], hparams)
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{:03d}.wav'.format(basenames[i])), hparams)


	def eval(self, text):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names))]
		input_lengths = [len(seq) for seq in seqs]
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}
		lf0s, mgcs, baps = self.session.run([self.model.lf0_outputs, self.model.mgc_outputs, self.bap_outputs], feed_dict=feed_dict)

		wavs = []
		for i, _ in enumerate(lf0s):
			wavs.append(audio.synthesize(lf0s[i], mgcs[i], baps[i], hparams))
		return np.concatenate(wavs)


	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs])

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)
