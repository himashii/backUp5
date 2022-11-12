from scipy.io import wavfile
import noisereduce as nr

rate, data = wavfile.read("input.wav")

reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("smooth.wav", rate, reduced_noise)