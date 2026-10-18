# Webcam Controller – Phase-2-Benchmark

Ausgabefreier Benchmark für lokale Gesichts- und Kopfgestenerkennung unter
Windows 10/11. Dieses Projekt erzeugt **keine** Tastatur-, Maus-, OpenTrack-
oder Spielausgaben.

## Umfang

- Kameraaufnahme mit Latest-Frame-Puffer (1080p60 bevorzugt, 720p30 Fallback)
- MediaPipe Face Landmarker als erstes Backend
- anatomisch getrennte Lid-, Mund- und Brauensignale
- Yaw/Pitch/Roll zur poseabhängigen Sperrung von Augengesten
- ausschließlich zeitbasierte Gestenlogik in Millisekunden
- Messung von Capture-, Inferenz-, Gesten- und hypothetischer Mapping-Latenz
- rollende Median-, p95- und p99-Werte
- synthetischer Testmodus ohne Kamera und Modell

## Bewusste Ausschlüsse

- kein `SendInput`
- kein Output-Broker
- kein OpenTrack-UDP
- keine Spiele oder Anti-Cheat-Tests
- keine unbemerkte Speicherung von Kamerabildern oder Videos
- Gesichts-ROI-Aufnahmen nur nach ausdrücklichem Start des lokalen Opt-in-Modus

## Installation

Voraussetzung ist 64-Bit-Python 3.11. Danach im Projektverzeichnis:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Das Face-Landmarker-Modell wird bewusst nicht ungefragt heruntergeladen.
Download aus der offiziellen MediaPipe-Quelle:

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py
```

## Schnelltest ohne Kamera

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark synthetic --seconds 10
.\.venv\Scripts\python.exe -m pytest
```

## Kamera-Benchmark

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark camera `
  --model models\face_landmarker.task `
  --width 1920 --height 1080 --fps 60 `
  --analysis-rate 60
```

Kameras unterscheiden sich bei der Format-Aushandlung. Standard ist
`--fourcc auto`; `--fourcc mjpg` darf nur verwendet werden, wenn die gemessene
Capture-Rate den Modus tatsächlich bestätigt. Die von einem Treiber gemeldete
Soll-FPS allein gilt nicht als Messwert.

### Getestetes Referenzprofil: Innomaker U20CAM-4M

Das am 18. Oktober 2026 geprüfte 1080p60-Profil verwendet DirectShow, MJPEG
und eine manuelle Belichtung von `-6` (ungefähr 1/64 Sekunde):

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark camera `
  --model models\face_landmarker.task `
  --camera 0 --capture-backend dshow --fourcc mjpg `
  --width 1920 --height 1080 --fps 60 `
  --exposure -6 --analysis-rate 60
```

Gemessen wurden 60,02 Capture-fps, 57,35 Analysen/s, 100 % Gesichtserkennung
sowie 10,36 ms Median- und 18,46 ms p95-Softwarelatenz. Der passende
Belichtungswert hängt von Kamera und Beleuchtung ab. Ohne `--exposure -6`
begrenzte das getestete Gerät die Rate auf ungefähr 25 fps. Der Bericht weist
angefordertes und tatsächlich verwendetes FourCC sowie die Belichtung aus.

### Getestetes Kameraprofil: NexiGo N60

Die NexiGo N60 unterstützt laut Hersteller maximal 1920x1080 bei 30 fps.
Unter Windows kann der Treiber bei einer Anforderung von 60 fps trotzdem
60 fps als Sollwert zurückmelden; maßgeblich ist daher `measured_rate`.
Das am 1. Oktober 2026 geprüfte Profil ist:

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark camera `
  --model models\face_landmarker.task `
  --camera 0 --capture-backend dshow `
  --width 1920 --height 1080 --fps 30 `
  --analysis-rate 30
```

Gemessen wurden 30,02 Capture-fps und 29,89 Analysen/s. Eine Anforderung von
1280x720 bei 60 fps lieferte mit dem getesteten Gerät nur etwa 9,9 Capture-fps
und ist deshalb kein empfohlenes Profil. Die projektweite 1080p60-Referenz
bleibt für Kameras bestehen, die diesen Modus tatsächlich liefern.

## Geführter Gestentest

Nach bestandenem Leistungstest führt dieser Modus mit sichtbaren Hinweisen
durch neutrale Mimik, normales Blinzeln, einseitiges Augenschließen, Mund,
Brauen und Kopfpose. Er erzeugt weiterhin keinerlei Systemeingaben:

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark guided `
  --model models\face_landmarker.task `
  --camera 0 --capture-backend dshow `
  --width 1920 --height 1080 --fps 30 `
  --analysis-rate 30 `
  --export results\n60-guided.json
```

Jeder Abschnitt wartet auf die Leertaste; danach erscheinen drei Sekunden
Vorbereitung. Während `MESSUNG`
wird die angezeigte Geste gehalten beziehungsweise beim Blinktest mehrmals
normal geblinzelt. `Esc` bricht ab. Der optionale JSON-Export enthält nur
aggregierte Signalwerte und keine Einzelbilder, Videos oder biometrischen
Landmark-Datensätze.

Falls 60 Analysen/s nicht gehalten werden, kann die Analyse begrenzt werden,
während die Kamera weiter mit 60 fps läuft:

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark camera `
  --model models\face_landmarker.task `
  --width 1920 --height 1080 --fps 60 `
  --analysis-rate 30
```

Mit `--no-preview` wird der Einfluss der Vorschau auf die Messung entfernt.
`Esc` beendet die Vorschau. Ergebnisse werden standardmäßig nur im Terminal
angezeigt und nicht dauerhaft gespeichert. Ein JSON-Export erfolgt nur mit
`--export <pfad>`.

## Messdefinitionen

- `capture_to_inference_ms`: Frame-Ankunft bis Ende der Erkennung
- `software_ms`: Frame-Ankunft bis Ende des hypothetischen Mappings
- `frame_age_at_inference_ms`: Alter des Frames zu Inferenzbeginn
- `inference_ms`: reine Backend-Laufzeit
- `gesture_ms`: Signalaufbereitung und Gestenzustandsautomaten
- `mapping_ms`: ausgabefreies Mapping

Ziel auf i5-8250U/Ryzen 5 3500U:

- mindestens 30 Analysen/s, 60/s zu prüfen
- Softwarelatenz Median <= 30 ms, p95 <= 50 ms
- möglichst <= 20–25 % Gesamt-CPU
- Arbeitsspeicher möglichst < 500 MB

## Datenschutz

Kamerabilder werden nur im Arbeitsspeicher verarbeitet. Exporte enthalten
ausschließlich aggregierte technische Messwerte und keine Bilder,
Fenstertitel, Namen oder personenbezogenen Pfade.

Der Benchmark pinnt MediaPipe derzeit auf 0.10.21. Beim Test von 0.10.35 wurde
ein ausgehender Clearcut-Metrikversuch beobachtet; eine offizielle Opt-out-
Option ist nicht dokumentiert. Neuere Wheels werden erst nach einem separaten
Netzwerk-Audit zugelassen.

## Lokaler Aufnahme- und Replaymodus

Dieser Modus ist ausdrücklich opt-in und speichert sensible Gesichtsdaten.
Standardmäßig wird nur eine quadratische Gesichts-ROI mit rotem sichtbarem
Aufnahmeindikator erfasst; das vollständige Kamerabild wird nicht gespeichert.
Datensätze unter `datasets/` sowie AVI-Dateien sind von Git ausgeschlossen.

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark record `
  --model models\face_landmarker.task `
  --output datasets\baseline-n60 `
  --camera 0 --capture-backend dshow `
  --width 1920 --height 1080 --fps 30 `
  --trials-per-gesture 10 `
  --auto-advance
```

Mit `--auto-advance` ist nur zum Start einmal die Leertaste erforderlich.
Zwischen Versuchen läuft danach automatisch ein Countdown. Beim Wechsel der
geforderten Aktion wechselt zusätzlich die Farbe der Anweisung; die längere
Wechselpause lässt sich mit `--group-change-seconds` einstellen.

Für getrennte Mundversuche gibt es einen eigenen automatischen Ablauf:

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark record-mouth `
  --model models\face_landmarker.task `
  --output datasets\mouth-1080p60 `
  --camera 0 --capture-backend dshow `
  --width 1920 --height 1080 --fps 60 `
  --fourcc MJPG --exposure -6
```

Dieser zeichnet zunächst zehn neutrale Abschnitte und anschließend zehn
einmalige Mundöffnungen auf. Die Offline-Auswertung prüft Kombinationen aus
`jawOpen`-Schwelle, Mindestdauer in Millisekunden und kurzer Lückenhysterese.

Der Zielordner muss leer oder noch nicht vorhanden sein und wird nie
überschrieben. `face_roi.avi` enthält MJPEG-Einzelbilder; `manifest.json`
enthält relative Capture-Zeitstempel, Sollgeste und Versuchsnummer sowie einen
SHA-256-Hash des Videos. Es werden keine Namen oder absoluten Pfade in das
Manifest geschrieben.

Wiederholbare Offline-Auswertung desselben Datensatzes:

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark replay `
  --model models\face_landmarker.task `
  --dataset datasets\baseline-n60 `
  --sweep `
  --export results\baseline-n60-replay.json
```

Für einen Negativtest bei Kopfrotation zeichnet `record-pose` fünf getrennt
beschriftete Abschnitte (neutral, links, rechts, oben und unten) auf. Während
jeder Position wird nur natürlich geblinzelt; absichtliche einseitige
Augengesten gehören nicht in diesen Datensatz.

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark record-pose `
  --model models\face_landmarker.task `
  --output datasets\negative-pose-n60 `
  --camera 0 --capture-backend dshow `
  --width 1920 --height 1080 --fps 30 `
  --seconds-per-pose 12
```

Ein kontinuierlicher Kalibrierungssweep vergleicht mehrere relative
Pose-Grenzen und deren Einfluss auf falsche Augengesten:

```powershell
.\.venv\Scripts\python.exe -m webcam_benchmark record-pose-sweep `
  --model models\face_landmarker.task `
  --output datasets\pose-sweep-n60 `
  --camera 0 --capture-backend dshow `
  --width 1920 --height 1080 --fps 30 `
  --seconds-per-step 18
```

Replay eignet sich für Erkennungsraten und den relativen Vergleich von
Algorithmen. Physische Kameralatenz, reale Capture-Rate und Kameraausfälle
müssen weiterhin live geprüft werden.

`--sweep` prüft EAR-Geschlossenheit, seitengetrennte Asymmetrie,
Mindestdauer, ausschließlich kausale Glättung und Gap-Hysterese. Werte unter
90 ms werden zwar explorativ ausgewiesen, aber nicht als freigabefähige
Kandidaten gerankt. Ein gutes Ergebnis auf demselben Datensatz ersetzt keine
Validierung unter unabhängigen Aufnahmebedingungen.

## Stand der automatischen 1080p60-Aufnahmen

Die am 18. Oktober 2026 automatisch aufgenommenen Innomaker-Datensätze
bestätigten den Workflow bei 1080p60. Im Augen-Datensatz wurden bei der
verbindlichen Mindestdauer von 90 ms 7/10 linke und 10/10 rechte Gesten
erkannt; 10/10 normale Blinzler blieben ohne Fehlauslösung. Der kontrollierte
Mund-Datensatz erreichte 10/10 Mundöffnungen und 0/10 Fehlauslösungen in
neutralen Versuchen. Das ist noch keine Produktivfreigabe: Sprechen, Gähnen,
natürliche Mundbewegungen sowie variierende Beleuchtung und Kopfrotation
fehlen als Negativ- und Robustheitstests.
