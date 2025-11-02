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

En la gráfica se puede observar, que ambas curvas, van de forma decreciente, tanto la frecuencia media, como la frecuencia mediana, van disminuyendo progresivamente con el número de contracciones, lo que significa reducción del contenido en altas frecuencias conforme aumenta la fatiga muscular.
La reducción de la frecuencia media y mediana es un indicador típico de fatiga muscular, debido a:
1. Disminución en la velocidad de conducción de las fibras musculares.
2. Aumento relativo de componentes de baja frecuencia en el espectro.

# PARTE B-CAPTURA DE LA SEÑAL DEL PACIENTE

Para la adquisición de la señal EMG real, se colocaron electrodos de superficie sobre el músculo bíceps braquial del brazo derecho, con un electrodo de referencia en la muñeca. Nuestra compañera realizó cinco contracciones voluntarias isométricas de flexión del antebrazo contra resistencia, cada una de aproximadamente 1 segundo de duración. Entre contracciones se dejó un breve periodo de relajación para permitir la recuperación parcial del músculo.

+ **Valores de la señal adquirida**

```python
from google.colab import files
up = files.upload()

import pandas as pd
df = pd.read_csv(next(iter(up.keys())), sep='\t')
df.head()
```
| Tiempo (s) | Amplitud (V) |
|-------------|--------------|
| 0.0000      | 0.000012     |
| 0.0005      | 0.002987     |
| 0.0010      | -0.002741    |
| 0.0015      | -0.008906    |
| 0.0020      | -0.004547    |

+ **Graficamos la señal**
```python
import numpy as np
import matplotlib.pyplot as plt

t = df['Tiempo (s)'].to_numpy()
x = df['Amplitud (V)'].to_numpy()
fs = int(round(1.0 / (t[1]-t[0])))

plt.figure(figsize=(10,4))
plt.plot(t, x, linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title("Señal EMG real")
plt.grid(True)
plt.show()

print(f"fs = {fs} Hz | muestras = {len(x)} | duración ≈ {t[-1]-t[0]:.2f} s")
```
<img width="684" height="312" alt="image" src="https://github.com/user-attachments/assets/467c9745-5fdf-4e3d-9093-60d4a9e71ee6" />
+ **fs = 2000 Hz**
+ **muestras = 10000**
+ **duración ≈ 5.00 s**

Filtramos la señal:
```python
from scipy.signal import butter, filtfilt

def bandpass(sig, fs, low=20, high=450, order=4):
    b, a = butter(order, [low, high], btype='bandpass', fs=fs)
    return filtfilt(b, a, sig)

xf = bandpass(x, fs, 20, 450, order=4)

plt.figure(figsize=(10,4))
plt.plot(t, xf, color='maroon', linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title("Señal EMG real filtrada 20–450 Hz")
plt.grid(True)
plt.show()
```
<img width="684" height="311" alt="image" src="https://github.com/user-attachments/assets/869c3ddf-a5f4-4f6c-b852-a1b5877113e0" />

+ **Dividimos la señal en el número de contracciones realizadas**
```python
from scipy.signal import find_peaks

# Rectificar y RMS móvil (~50 ms)
rect = np.abs(xf)
win = max(1, int(0.05 * fs))            # ventana 50 ms
rms = np.sqrt(np.convolve(rect**2, np.ones(win)/win, mode='same'))

# Umbral: media + 1.5*std (ajustable)
thr = rms.mean() + 1.5*rms.std()

# Picos en la envolvente (al menos 0.6 s entre contracciones)
peaks, _ = find_peaks(rms, height=thr, distance=int(0.6*fs))
peaks = peaks[:5]  # nos quedamos con los 5 primeros si hay más

# Visual de validación
plt.figure(figsize=(10,4))
plt.plot(t, rms, label='RMS (envolvente)')
plt.hlines(thr, t[0], t[-1], colors='r', linestyles='--', label='Umbral')
plt.plot(t[peaks], rms[peaks], 'ko', label='Contracciones detectadas')
plt.xlabel("Tiempo [s]"); plt.ylabel("RMS [V]")
plt.title("Detección automática de contracciones")
plt.grid(True); plt.legend(); plt.show()

print("Índices de picos:", peaks)
```
<img width="683" height="313" alt="image" src="https://github.com/user-attachments/assets/038f26a9-bb36-4877-afcd-4cc803c43920" />
+ **Índices de picos: [ 929 3090 4915 7091 9003]**

```python
import matplotlib.pyplot as plt
import numpy as np

# Partimos de: t, xf (señal filtrada 20–450 Hz), rms, peaks
plt.figure(figsize=(12,4))

# Señal filtrada en gris claro
plt.plot(t, xf, color='0.7', linewidth=0.8, label='Señal EMG filtrada (20–450 Hz)')
# Envolvente RMS (energía)
plt.plot(t, rms, color='darkblue', linewidth=1.5, label='Envolvente RMS (50 ms)')
# Umbral
plt.hlines(thr, t[0], t[-1], color='crimson', linestyles='--', label='Umbral')
# Picos/contracciones
plt.plot(t[peaks], rms[peaks], 'ko', label='Contracciones detectadas')

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud / RMS [V]')
plt.title('Detección automática de contracciones en EMG')
plt.grid(True, alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```
<img width="950" height="312" alt="image" src="https://github.com/user-attachments/assets/7d58961b-8ceb-4755-a7af-ff4f3338c5eb" />

Los segmentos de la señal fueron:
```python
segmentos = []
win_sec = 0.35   # medio ancho por lado del pico  (total ≈ 0.7 s)
L = int(win_sec * fs)

for p in peaks:
    i0 = max(p - L, 0)
    i1 = min(p + L, len(xf))
    segmentos.append(xf[i0:i1])

print(f"Segmentos obtenidos: {len(segmentos)} (longitud media ≈ {np.mean([len(s) for s in segmentos])/fs:.2f} s)")
```
+ *Segmentos obtenidos: 5 (longitud media=0.70s)*

+**Calculamos la frecuencia media y frecuencia mediana**
```python
from scipy.signal import windows
import pandas as pd

fm_list, fmed_list = [], []

for seg in segmentos:
    w = windows.hann(len(seg))
    segw = seg * w
    N = len(segw)
    f = np.fft.rfftfreq(N, d=1/fs)
    Pxx = np.abs(np.fft.rfft(segw))**2
    den = Pxx.sum() if Pxx.sum()!=0 else 1.0

    fm = (f*Pxx).sum()/den
    Pc = np.cumsum(Pxx)
    fmed = f[np.searchsorted(Pc, 0.5*Pc[-1])]

    fm_list.append(float(fm))
    fmed_list.append(float(fmed))

res = pd.DataFrame({
    "Contracción": np.arange(1, len(segmentos)+1),
    "Frecuencia Media (Hz)": np.round(fm_list, 2),
    "Frecuencia Mediana (Hz)": np.round(fmed_list, 2)
})
res
```
Obtenemos los siguientes resultados:

| Contracción | Frecuencia Media (Hz) | Frecuencia Mediana (Hz) |
|--------------|------------------------|--------------------------|
| 1            | 118.21                 | 128.57                   |
| 2            | 108.58                 | 124.29                   |
| 3            | 90.36                  | 94.29                    |
| 4            | 79.23                  | 77.14                    |
| 5            | 75.50                  | 70.00                    |

+ **Evolución de las frecuencias por contracción**
```python
xidx = res["Contracción"].to_numpy()
plt.figure(figsize=(8,5))
plt.plot(xidx, res["Frecuencia Media (Hz)"], 'o-', label="Frecuencia Media")
plt.plot(xidx, res["Frecuencia Mediana (Hz)"], 's-', label="Frecuencia Mediana")
plt.xlabel("Contracción")
plt.ylabel("Frecuencia [Hz]")
plt.title("Evolución de las frecuencias por contracción")
plt.grid(True); plt.legend(); plt.show()
```
<img width="554" height="377" alt="image" src="https://github.com/user-attachments/assets/3d2816b0-8b60-489b-824d-82e9a3a7790f" />

+ **FFT**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Asegúrate de tener 'segmentos' (lista con 5 arrays)
seg1 = segmentos[0]
seg5 = segmentos[-1]

def fft_pxx(seg, fs):
    w = windows.hann(len(seg))
    X = np.fft.rfft(seg * w)
    f = np.fft.rfftfreq(len(seg), d=1/fs)
    Pxx = (np.abs(X)**2)
    # normalizar para comparar forma
    Pxx = Pxx / (Pxx.max() + 1e-12)
    return f, Pxx

f1, P1 = fft_pxx(seg1, fs)
f5, P5 = fft_pxx(seg5, fs)

plt.figure(figsize=(8,5))
plt.plot(f1, P1, label='Contracción 1', linewidth=1.5)
plt.plot(f5, P5, label='Contracción 5', linewidth=1.5)
plt.xlim(0, 200)               # rango típico ilustrativo
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia normalizada')
plt.title('Espectro (FFT) 1ª vs 5ª contracción')
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
```
<img width="632" height="391" alt="image" src="https://github.com/user-attachments/assets/9bdf5928-90aa-47d6-9bf7-fd73996cf2ae" />














