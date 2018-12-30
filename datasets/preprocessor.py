import glob, os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from datasets import audio


def build_from_path(hparams, input_dirs, lf0_dir, mgc_dir, bap_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		trn_files = glob.glob(os.path.join(input_dir, 'biaobei_48000', '*.trn'))
		for trn in trn_files:
			with open(trn) as f:
				basename = trn[:-4]
				wav_file = basename + '.wav'
				wav_path = wav_file
				basename = basename.split('/')[-1]
				text = f.readline().strip()
				futures.append(executor.submit(partial(_process_utterance, lf0_dir, mgc_dir, bap_dir, basename, wav_path, text, hparams)))
				index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(lf0_dir, mgc_dir, bap_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, hparams)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	# feature extraction
	f0, sp, ap = audio.feature_extract(wav, hparams)
	n_frames = len(f0)
	if n_frames > hparams.max_frame_num:
		return None
	
	# feature normalization
	lf0 = audio.f0_normalize(f0, hparams)
	mgc = audio.sp_normalize(sp, hparams)
	bap = audio.ap_normalize(ap, hparams)
	
	lf0_file = 'lf0-{}.npy'.format(index)
	mgc_file = 'mgc-{}.npy'.format(index)
	bap_file = 'bap-{}.npy'.format(index)
	np.save(os.path.join(lf0_dir, lf0_file), lf0, allow_pickle=False)
	np.save(os.path.join(mgc_dir, mgc_file), mgc, allow_pickle=False)
	np.save(os.path.join(bap_dir, bap_file), bap, allow_pickle=False)

	# Return a tuple describing this training example
	return (lf0_file, mgc_file, bap_file, n_frames, n_frames, text)
