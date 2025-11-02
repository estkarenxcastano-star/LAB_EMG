# LABORATORIO 4
## EMG
### OBJETIVO
Analizar señales electromiográficas (EMG) mediante filtrado y análisis espectral para detectar la fatiga muscular y comparar el comportamiento entre señales reales y emuladas.

# PARTE A-CAPTURA DE LA SEÑAL EMULADA

## LIBERERIAS
Las librerias que implementamos fueron las siguientes:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import find_peaks
```
Para capturar la señal utilizamos el generador de señales fisiologicas del laboratorio y la capturamos por medio del DAQ, obteniendo la siguiente señal:
```python
from google.colab import files
up = files.upload()

import pandas as pd
df = pd.read_csv(next(iter(up.keys())), sep='\t')
df.head()
```
| Tiempo (s) | Amplitud (V) |
|-------------|--------------|
| 0.0000      | -0.000988    |
| 0.0005      | -0.000700    |
| 0.0010      | -0.004206    |
| 0.0015      | -0.025752    |
| 0.0020      | -0.057613    |

Código para gráficar la señal:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(df['Tiempo (s)'], df['Amplitud (V)'], linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title("Señal Simulada")
plt.grid(True)
plt.show()
```
<img width="686" height="314" alt="image" src="https://github.com/user-attachments/assets/fc02d42b-ec97-456f-aa35-556d464a55bd" />

+ **Segmentamos la señal en 5 contracciones**
```python
import numpy as np

t = df['Tiempo (s)'].to_numpy()
x = df['Amplitud (V)'].to_numpy()

fs = int(round(1.0 / (t[1]-t[0])))
duracion = t[-1] - t[0]
print(f"fs = {fs} Hz, duración ≈ {duracion:.2f} s, muestras = {len(x)}")
```
Obtenemos:
+ *fs = 2000 Hz*
+ *duración ≈ 5.00 s*
+ *muestras = 10000*

Filtramos la señal:
```python
from scipy.signal import butter, filtfilt

def bandpass(sig, fs, low=20, high=450, order=4):
    b, a = butter(order, [low, high], btype='bandpass', fs=fs)
    return filtfilt(b, a, sig)

x_f = bandpass(x, fs, 20, 450, order=4)

plt.figure(figsize=(10,4))
plt.plot(t, x_f, linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title("Señal simulada y filtrada 20–450 Hz")
plt.grid(True)
plt.show()
```
<img width="685" height="311" alt="image" src="https://github.com/user-attachments/assets/bfe3daef-b91e-4032-ae71-bcec7eb09f25" />

Ahora si, segmentamos la señal:
```python
segmentos = np.array_split(x_f, 5)  # usa x_f; si no filtraste, cambia a x

for i, seg in enumerate(segmentos):
    print(f"Contracción {i+1}: {len(seg)} muestras (~{len(seg)/fs:.2f} s)")
```
| Contracción | Número de muestras | Duración aproximada |
|--------------|--------------------|----------------------|
| 1            | 2000               | ~1.00 s              |
| 2            | 2000               | ~1.00 s              |
| 3            | 2000               | ~1.00 s              |
| 4            | 2000               | ~1.00 s              |
| 5            | 2000               | ~1.00 s              |

+ **Calculamos la frecuencia media y frecuencia mediana**
```python
from scipy.signal import windows
import pandas as pd

freq_media, freq_mediana = [], []

for seg in segmentos:
    w = windows.hann(len(seg))
    segw = seg * w
    N = len(segw)
    f = np.fft.rfftfreq(N, d=1/fs)
    Pxx = np.abs(np.fft.rfft(segw))**2

    den = Pxx.sum() if Pxx.sum() != 0 else 1.0
    fm = np.sum(f * Pxx) / den

    Pc = np.cumsum(Pxx)
    idx_med = int(np.searchsorted(Pc, 0.5 * Pc[-1]))
    fmed = float(f[min(idx_med, len(f)-1)])

    freq_media.append(float(fm))
    freq_mediana.append(float(fmed))

res = pd.DataFrame({
    "Contracción": [1,2,3,4,5],
    "Frecuencia Media (Hz)": np.round(freq_media, 2),
    "Frecuencia Mediana (Hz)": np.round(freq_mediana, 2)
})
res
```
Como resultado obtenemos:
| Contracción | Frecuencia Media (Hz) | Frecuencia Mediana (Hz) |
|--------------|------------------------|--------------------------|
| 1            | 219.47                 | 228.0                    |
| 2            | 212.11                 | 217.0                    |
| 3            | 208.24                 | 198.0                    |
| 4            | 231.96                 | 245.0                    |
| 5            | 215.19                 | 206.0                    |

+ **Evolución de las frecuencias por contracción**
```python
xidx = np.arange(1, 6)
plt.figure(figsize=(8,5))
plt.plot(xidx, res["Frecuencia Media (Hz)"], marker='o', label="Frecuencia Media")
plt.plot(xidx, res["Frecuencia Mediana (Hz)"], marker='s', label="Frecuencia Mediana")
plt.xlabel("Contracción")
plt.ylabel("Frecuencia [Hz]")
plt.title("Evolución de las frecuencias por contracción")
plt.grid(True)
plt.legend()
plt.show()
```
<img width="552" height="376" alt="image" src="https://github.com/user-attachments/assets/ce6e3a34-0fd2-4ae0-a697-6b990be1a0ef" />













