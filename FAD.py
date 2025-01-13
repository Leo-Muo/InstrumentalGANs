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
    
    

===============================
# Training Function
def train(generator, discriminator, dataloader, g_optimizer, d_optimizer, epochs, device):
    generator.train()
    discriminator.train()
    g_loss_item = 0
    d_loss_item = 0

    for epoch in range(epochs):
        for i, (vocal, instrumental) in enumerate(dataloader):
            vocal, instrumental = vocal.to(device), instrumental.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_data = torch.cat((vocal, instrumental), dim=1)
            fake_data = torch.cat((vocal, generator(vocal)), dim=1)

            real_loss = adversarial_loss(discriminator(real_data), torch.ones_like(discriminator(real_data)))
            fake_loss = adversarial_loss(discriminator(fake_data), torch.zeros_like(discriminator(fake_data)))
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            fake_data, mu, logvar = generator(vocal)
            g_loss_adv = adversarial_loss(discriminator(torch.cat((vocal, fake_data), dim=1)), torch.ones_like(discriminator(torch.cat((vocal, fake_data), dim=1))))
            g_loss_rec = reconstruction_loss(fake_data, instrumental)
            g_loss = g_loss_adv + g_loss_rec

            g_loss.backward()
            g_optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
                
            g_loss_item = g_loss.item()
            d_loss_item = d_loss.item()
            
                    
    return d_loss_item, g_loss_item


========================

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)  # Output single-channel feature map
        )
    
    def forward(self, x):
        return self.model(x)