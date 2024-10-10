import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch, iirnotch

# 1. Chargement des données
file_path = 'C:/Users/shait/Downloads/transfer_8311502_files_e5150916/Exercice_EyesOpen_EyesClosed/Exercice_EyesOpen_EyesClosed/Subject1-1stTime/OpenBCI-RAW-2021-04-08_11-19-46.csv'
data = pd.read_csv(file_path, skiprows=4)

eeg_data = data.iloc[-75000:, 1:9]  # Colonnes 1 à 8 contiennent les canaux EEG
eeg_data = eeg_data.apply(pd.to_numeric, errors='coerce')  # Conversion en numérique 

sampling_rate = 250  # Hz
window_length = 250  # Fenêtres de 1 seconde
theta_band, alpha_band = (4, 8), (8, 12)

# 2. Filtrage du signal
def apply_filters(data, lowcut=4, highcut=60, fs=250, order=5, notch_freq=50, Q=30):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b_bandpass, a_bandpass = butter(order, [low, high], btype='band')
    w0 = notch_freq / nyquist
    b_notch, a_notch = iirnotch(w0, Q)
    data = lfilter(b_bandpass, a_bandpass, data)
    return lfilter(b_notch, a_notch, data)

# 3. Calcul des puissances Theta et Alpha
def calculate_band_power(data, band, fs=250):
    freqs, psd = welch(data, fs, nperseg=window_length)
    return np.sum(psd[(freqs >= band[0]) & (freqs <= band[1])])

# 4. Appliquer les filtres et calculer les puissances relatives
theta_powers, alpha_powers = [], []
for channel in eeg_data:
    filtered_channel = apply_filters(eeg_data[channel].values)
    theta_power = [calculate_band_power(filtered_channel[i:i + window_length], theta_band)
                   for i in range(0, len(filtered_channel), window_length)]
    alpha_power = [calculate_band_power(filtered_channel[i:i + window_length], alpha_band)
                   for i in range(0, len(filtered_channel), window_length)]
    theta_powers.append(theta_power)
    alpha_powers.append(alpha_power)

# Convertir en array et transposer pour avoir 100 lignes (échantillons) et 8 colonnes (canaux)
theta_powers, alpha_powers = np.array(theta_powers).T, np.array(alpha_powers).T

# Vérifier les dimensions avant sauvegarde
print(f"Shape of relative_alpha (after transpose): {alpha_powers.shape}")
print(f"Shape of relative_theta (after transpose): {theta_powers.shape}")

# 5. Calcul des puissances relatives
total_power = theta_powers + alpha_powers
relative_theta = theta_powers / total_power
relative_alpha = alpha_powers / total_power

# 6. Génération du graphique et sauvegarde dans un fichier
time = np.arange(len(relative_alpha))
plt.plot(time, relative_alpha[:, 0], label='Alpha', color='blue')
plt.plot(time, relative_theta[:, 0], label='Theta', color='orange')
plt.title('Relative Alpha and Theta Power Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Relative Power')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Sauvegarder le graphique dans un fichier image
plt.savefig('alpha_theta_power_over_time.png')  # Sauvegarde sous forme d'image

# 7. Sauvegarder les puissances Alpha et Theta dans des fichiers CSV
np.savetxt('alpha_data.csv', relative_alpha, delimiter=',')
np.savetxt('theta_data.csv', relative_theta, delimiter=',')
