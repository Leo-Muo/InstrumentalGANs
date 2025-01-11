import torch
import numpy as np
import librosa
from openl3 import get_embedding




import tensorflow as tf
from vggish import vggish_input, vggish_postprocess, vggish_slim

class VGGishEmbeddingModel:
    def __init__(self, model_path='vggish_model.ckpt', pca_params_path='vggish_pca_params.npz'):
        self.model_path = model_path
        self.pca_params_path = pca_params_path
        self.session = tf.compat.v1.Session()

        # Load VGGish model
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(self.session, self.model_path)

        # Load PCA postprocessor
        self.pproc = vggish_postprocess.Postprocessor(self.pca_params_path)

    def extract_features(self, audio, sr=16000):
        # VGGish expects 16 kHz mono audio
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if len(audio.shape) > 1:  # Convert stereo to mono
            audio = librosa.to_mono(audio)

        # Convert audio to examples (log-mel spectrogram)
        examples_batch = vggish_input.waveform_to_examples(audio, sr=16000)
        [embedding_batch] = self.session.run(
            ['vggish/embeddings:0'],
            feed_dict={'vggish/input_features:0': examples_batch}
        )

        # Apply PCA postprocessing
        return self.pproc.postprocess(embedding_batch)

# Assume you have a function to compute FAD
def compute_fad(real_embeddings, generated_embeddings):
    mu_real = np.mean(real_embeddings, axis=0)
    mu_generated = np.mean(generated_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)
    sigma_generated = np.cov(generated_embeddings, rowvar=False)

    # Compute the Fréchet Distance
    diff = mu_real - mu_generated
    covmean, _ = np.linalg.sqrtm(sigma_real.dot(sigma_generated), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = np.sum(diff**2) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fad

def validate(generator, dataloader, device, embedding_model):
    generator.eval()
    snr_list, lsd_list = [], []
    real_embeddings, predicted_embeddings = [], []

    with torch.no_grad():
        for vocal, instrumental in dataloader:
            vocal, instrumental = vocal.to(device), instrumental.to(device)
            predicted = generator(vocal)

            # Calculate SNR
            signal_power = torch.sum(instrumental**2)
            noise_power = torch.sum((instrumental - predicted)**2)
            snr = 10 * torch.log10(signal_power / noise_power)
            snr_list.append(snr.item())

            # Calculate LSD
            instrumental_spec = librosa.amplitude_to_db(torch.squeeze(instrumental.cpu()).numpy(), ref=np.max)
            predicted_spec = librosa.amplitude_to_db(torch.squeeze(predicted.cpu()).numpy(), ref=np.max)
            lsd = np.mean(np.sqrt(np.mean((instrumental_spec - predicted_spec)**2, axis=-1)))
            lsd_list.append(lsd)

            # Compute embeddings for FAD
            real_embedding = embedding_model.extract_features(instrumental.cpu().numpy())
            generated_embedding = embedding_model.extract_features(predicted.cpu().numpy())
            
            real_embeddings.append(real_embedding)
            predicted_embeddings.append(generated_embedding)

    avg_snr = np.mean(snr_list)
    avg_lsd = np.mean(lsd_list)

    # Compute FAD
    real_embeddings = np.concatenate(real_embeddings)
    predicted_embeddings = np.concatenate(predicted_embeddings)
    fad_score = compute_fad(real_embeddings, predicted_embeddings)

    print(f"Validation Results - SNR: {avg_snr:.4f}, LSD: {avg_lsd:.4f}, FAD: {fad_score:.4f}")





============================



def validate(generator, dataloader, device):
    generator.eval()
    snr_list, lsd_list = [], []

    with torch.no_grad():
        for vocal, instrumental in dataloader:
            vocal, instrumental = vocal.to(device), instrumental.to(device)
            predicted = generator(vocal)

            # Calculate SNR
            signal_power = torch.sum(instrumental**2)
            noise_power = torch.sum((instrumental - predicted)**2)
            snr = 10 * torch.log10(signal_power / noise_power)
            snr_list.append(snr.item())

            # Calculate LSD
            instrumental_spec = librosa.amplitude_to_db(torch.squeeze(instrumental.cpu()).numpy(), ref=np.max)
            predicted_spec = librosa.amplitude_to_db(torch.squeeze(predicted.cpu()).numpy(), ref=np.max)
            lsd = np.mean(np.sqrt(np.mean((instrumental_spec - predicted_spec)**2, axis=-1)))
            lsd_list.append(lsd)

    avg_snr = np.mean(snr_list)
    avg_lsd = np.mean(lsd_list)

    print(f"Validation Results - SNR: {avg_snr:.4f}, LSD: {avg_lsd:.4f}")