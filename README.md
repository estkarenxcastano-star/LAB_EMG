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

