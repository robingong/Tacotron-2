import librosa
import numpy as np
import pysptk
import pyworld as vocoder
import soundfile as sf
import tensorflow as tf

int16_max = 32768.0

def load_wav(path, hparams):
	wav, _ = sf.read(path)
	# rescale wav for unified measure for all clips
	return wav# / np.abs(wav).max() * hparams.rescale_max

def save_wav(wav, path, hparams):
	# wav = wav / np.abs(wav).max() * hparams.rescale_max
	sf.write(path, wav, hparams.sample_rate)

def trim_silence(wav, hparams):
	return librosa.effects.trim(wav, top_db= hparams.trim_top_db,
		frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]

def feature_extract(wav, hparams):
	return vocoder.wav2world(wav, hparams.sample_rate, hparams.fft_size, ap_depth=hparams.num_bap)

def synthesize(lf0, mgc, bap, hparams):
	lf0 = np.where(lf0 < 1, 0.0, lf0)
	f0 = f0_denormalize(lf0, hparams)
	sp = sp_denormalize(mgc, hparams)
	ap = ap_denormalize(bap, lf0, hparams)
	wav = vocoder.synthesize(f0, sp, ap, hparams.sample_rate)
	return wav

def f0_normalize(x, hparams):
	return np.log(np.where(x == 0.0, 1.0, x)).astype(np.float32)

def f0_denormalize(x, hparams):
	return np.where(x == 0.0, 0.0, np.exp(x.astype(np.float64)))

def sp_normalize(x, hparams):
	sp = int16_max * np.sqrt(x)
	return pysptk.sptk.mcep(sp.astype(np.float32), order=hparams.num_mgc - 1, alpha=hparams.mcep_alpha,
				maxiter=0, threshold=0.001, etype=1, eps=1.0E-8, min_det=0.0, itype=3)

def sp_denormalize(x, hparams):
	sp = pysptk.sptk.mgc2sp(x.astype(np.float64), order=hparams.num_mgc - 1,
				alpha=hparams.mcep_alpha, gamma=0.0, fftlen=hparams.fft_size)
	return np.square(sp / int16_max)

def ap_normalize(x, hparams):
	return x.astype(np.float32)

def ap_denormalize(x, lf0, hparams):
	for i in range(len(lf0)):
		x[i] = np.where(lf0[i] == 0, np.zeros(x.shape[1]), x[i])
	return x.astype(np.float64)
