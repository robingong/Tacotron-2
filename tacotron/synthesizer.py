import os
import wave
from datetime import datetime
import io
import numpy as np
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
		targets = tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets')
		with tf.variable_scope('model') as scope:
			self.model = create_model(model_name, hparams)
			if gta:
				self.model.initialize(inputs, input_lengths, targets, gta=gta)
			else:
				self.model.initialize(inputs, input_lengths)
			self.alignments = self.model.alignments
			self.mel_outputs = self.model.mel_outputs
			self.stop_token_prediction = self.model.stop_token_prediction

		self.gta = gta
		self._hparams = hparams
		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -(hparams.max_abs_value + .1)
		else:
			self._target_pad = -0.1

		log('Loading checkpoint: %s' % checkpoint_path)
		#Memory allocation on the GPU as needed
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		self.session = tf.Session(config=config)
		self.session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, basenames, out_dir, log_dir, mel_filenames):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}

		if self.gta:
			np_targets = [np.load(mel_filename) for mel_filename in mel_filenames]
			target_lengths = [len(np_target) for np_target in np_targets]
			padded_targets = self._prepare_targets(np_targets, self._hparams.outputs_per_step)
			feed_dict[self.model.mel_targets] = padded_targets.reshape(len(np_targets), -1, hparams.num_mels)

			mels, alignments = self.session.run([self.mel_outputs, self.alignments], feed_dict=feed_dict)
			mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)] #Take off the reduction factor padding frames for time consistency with wavenet
			assert len(mels) == len(np_targets)

		saved_mels_paths = []
		speaker_ids = []
		for i, mel in enumerate(mels):
			#Get speaker id for global conditioning (only used with GTA generally)
			speaker_id = '<no_g>'
			speaker_ids.append(speaker_id)

			# Write the spectrogram to disk
			# Note: outputs mel-spectrogram files and target ones have same names, just different folders
			mel_filename = os.path.join(out_dir, '{}.npy'.format(basenames[i]))
			np.save(mel_filename, mel.T, allow_pickle=False)
			saved_mels_paths.append(mel_filename)

			if log_dir is not None:
				#save wav (mel -> wav)
				wav = audio.inv_mel_spectrogram(mel.T, hparams)
				audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-mel.wav'.format(basenames[i])), hparams)

				#save alignments
				plot.plot_alignment(alignments[i], os.path.join(log_dir, 'plots/alignment-{}.png'.format(basenames[i])),
					info='{}'.format(texts[i]), split_title=True)

				#save mel spectrogram plot
				plot.plot_spectrogram(mel, os.path.join(log_dir, 'plots/mel-{}.png'.format(basenames[i])),
					info='{}'.format(texts[i]), split_title=True)

		return saved_mels_paths, speaker_ids

	def eval(self, batch):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in batch]
		input_lengths = [len(seq) for seq in seqs]
		seqs = self._prepare_inputs(seqs)
		feed_dict = {
			self.model.inputs: seqs,
			self.model.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}

		mels, stop_tokens = self.session.run([self.mel_outputs, self.stop_token_prediction], feed_dict=feed_dict)

		#Get Mel/Linear lengths for the entire batch from stop_tokens predictions
		target_lengths = self._get_output_lengths(stop_tokens)

		#Take off the batch wise padding
		mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
		assert len(mels) == len(batch)

		return np.concatenate(mels)

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs])

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		return np.stack([self._pad_target(t, self._round_up(max_len, alignment)) for t in targets])

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _get_output_lengths(self, stop_tokens):
		#Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
		output_lengths = [row.index(1) + 1 if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
		return output_lengths
