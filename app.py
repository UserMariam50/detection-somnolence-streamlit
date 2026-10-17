import os
import time
import tempfile
import subprocess
from collections import deque

import av
import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "face_landmarker.task"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

INITIAL_THRESHOLD = 0.15
BASELINE_FACTOR = 0.75
MIN_CLOSURE_SECONDS = 0.5
PERCLOS_WINDOW_SECONDS = 60
PERCLOS_ALERT = 15.0

# Calibration courte adaptée aux vidéos courtes
CALIBRATION_MIN_SECONDS = 0.5
CALIBRATION_MAX_SECONDS = 2.0
CALIBRATION_MIN_SAMPLES = 12
CALIBRATION_OPEN_EAR_MIN = INITIAL_THRESHOLD * 1.20
PERCLOS_MIN_VALID_COVERAGE = 0.75

# Seuil de validité de la baseline : en dessous, on
# considère que la calibration a appris des yeux mi-clos
MIN_VALID_BASELINE = 0.15

MIN_FACE_WIDTH_RATIO = 0.05

# Taille des vidéos de sortie
MAX_OUTPUT_WIDTH = 480
PREVIEW_WIDTH = 320

# Confirmation temporelle (secondes)
CLOSED_CONFIRM_SECONDS = 0.15
OPEN_CONFIRM_SECONDS = 0.10


# =========================================================
# UTILITAIRES EAR
# =========================================================

def euclidean_distance(p1, p2):
    return np.linalg.norm(
        np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
    )


def calculate_ear(landmarks, indices):
    p1, p2, p3, p4, p5, p6 = [
        landmarks[i] for i in indices
    ]

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal < 1e-8:
        return np.nan

    return (vertical_1 + vertical_2) / (2 * horizontal)


def create_landmarker():
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )


def format_closure_count(n):
    if n == 0:
        return "Aucune"
    if n == 1:
        return "1 fermeture"
    return f"{n} fermetures"


def _public_events(events):
    """Retire les clés internes (_open) des événements."""
    return [
        {k: v for k, v in ev.items() if not k.startswith("_")}
        for ev in events
    ]


# =========================================================
# DETECTEUR DE SOMNOLENCE
# =========================================================

class DrowsinessDetector:
    """État indépendant pour chaque vidéo ou session webcam."""

    def __init__(self, fps=None):
        self.landmarker = create_landmarker()

        self.reference_fps = float(fps) if fps else 30.0

        self.ear_history = deque(maxlen=5)
        self.baseline_samples = deque(maxlen=300)
        self.calibration_uncertain = False
        self.last_baseline_update = None
        self.baseline_start = None
        self.session_start = None

        self.samples = deque()
        self.events = []

        self.closed_start = None
        self.last_time = None
        self.last_timestamp_ms = -1
        self.last_ear = None
        self.last_left_ear = None
        self.last_right_ear = None
        self.last_threshold = INITIAL_THRESHOLD
        self.last_perclos = None
        self.last_status = "CALIBRATING"
        self.last_eye_state = "UNKNOWN"
        self.last_face_valid = False

        self._fallback_applied = False

        self.closed_since = None
        self.open_since = None

    # -----------------------------------------------------
    def threshold(self):
        # La baseline n'est utilisée que si suffisamment de mesures
        # plausibles d'yeux ouverts ont été recueillies.
        if self.calibration_ready():
            median_ear = float(np.median(self.baseline_samples))
            if median_ear >= MIN_VALID_BASELINE:
                return max(0.05, median_ear * BASELINE_FACTOR)
        return INITIAL_THRESHOLD

    def calibration_ready(self):
        if len(self.baseline_samples) < CALIBRATION_MIN_SAMPLES:
            return False
        median_ear = float(np.median(self.baseline_samples))
        return np.isfinite(median_ear) and median_ear >= MIN_VALID_BASELINE

    def in_calibration_phase(self):
        if self.session_start is None or self.last_time is None:
            return True
        elapsed = self.last_time - self.session_start
        if elapsed < CALIBRATION_MIN_SECONDS:
            return True
        if self.calibration_ready():
            return False
        return elapsed < CALIBRATION_MAX_SECONDS

    def _apply_fallback_baseline_if_needed(self):
        # Ne jamais fabriquer une baseline en dupliquant des valeurs
        # récentes : ces valeurs pourraient correspondre à des yeux fermés.
        if self.last_time is None or self.session_start is None:
            return
        if self.last_time - self.session_start >= CALIBRATION_MAX_SECONDS:
            self.calibration_uncertain = not self.calibration_ready()

    # -----------------------------------------------------
    def _reset_temporal_counters(self):
        self.closed_since = None
        self.open_since = None

    # -----------------------------------------------------
    def process(self, frame, timestamp_sec):
        height, width = frame.shape[:2]

        if self.session_start is None:
            self.session_start = timestamp_sec

        timestamp_ms = max(
            int(timestamp_sec * 1000),
            self.last_timestamp_ms + 1,
        )
        self.last_timestamp_ms = timestamp_ms

        if self.last_time is None:
            dt = 1 / self.reference_fps
        else:
            dt = float(np.clip(
                timestamp_sec - self.last_time, 0.001, 0.2
            ))
        self.last_time = timestamp_sec

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self.landmarker.detect_for_video(
            image, timestamp_ms
        )

        # -------- Pas de visage --------
        if not result.face_landmarks:
            self.closed_start = None
            self._reset_temporal_counters()
            self.samples.append({
                "t": timestamp_sec,
                "dt": dt,
                "valid": False,
                "prolonged": False,
            })
            self.last_ear = None
            self.last_left_ear = None
            self.last_right_ear = None
            self.last_eye_state = "UNKNOWN"
            self.last_face_valid = False
            self.last_status = "FACE NOT DETECTED"
            self._prune_samples(timestamp_sec)
            self.last_perclos = self.perclos()

            self._draw_overlay(
                frame, width, height,
                status="FACE NOT DETECTED",
                color=(0, 165, 255),
                face_box=None,
            )
            return frame, self.snapshot()

        landmarks = result.face_landmarks[0]

        # -------- Filtre taille minimale du visage --------
        xs_all = [p.x for p in landmarks]
        face_w_ratio = max(xs_all) - min(xs_all)

        if face_w_ratio < MIN_FACE_WIDTH_RATIO:
            self.closed_start = None
            self._reset_temporal_counters()
            self.samples.append({
                "t": timestamp_sec,
                "dt": dt,
                "valid": False,
                "prolonged": False,
            })
            self.last_ear = None
            self.last_left_ear = None
            self.last_right_ear = None
            self.last_eye_state = "UNKNOWN"
            self.last_face_valid = False
            self.last_status = "FACE TOO SMALL"
            self._prune_samples(timestamp_sec)
            self.last_perclos = self.perclos()

            self._draw_overlay(
                frame, width, height,
                status="FACE TOO SMALL",
                color=(0, 165, 255),
                face_box=None,
            )
            return frame, self.snapshot()

        # -------- EAR --------
        left_ear = calculate_ear(landmarks, LEFT_EYE)
        right_ear = calculate_ear(landmarks, RIGHT_EYE)

        if not np.isfinite([left_ear, right_ear]).all():
            self.samples.append({
                "t": timestamp_sec, "dt": dt,
                "valid": False, "prolonged": False,
            })
            self._prune_samples(timestamp_sec)
            self.last_perclos = self.perclos()
            self.last_status = "MEASUREMENT UNCERTAIN"
            self.last_eye_state = "UNKNOWN"
            self.last_face_valid = False
            self._reset_temporal_counters()
            self._draw_overlay(
                frame, width, height,
                status="MEASUREMENT UNCERTAIN",
                color=(0, 165, 255),
                face_box=None,
            )
            return frame, self.snapshot()

        raw_ear = (left_ear + right_ear) / 2
        self.ear_history.append(raw_ear)
        ear = float(np.median(self.ear_history))

        self.last_left_ear = float(left_ear)
        self.last_right_ear = float(right_ear)
        self.last_ear = ear
        self.last_face_valid = True

        # -------- Baseline / seuil --------
        in_calib = self.in_calibration_phase()

        if in_calib:
            # Écarter les valeurs trop basses, probablement liées à une
            # fermeture ou à une ouverture incomplète des yeux.
            if ear >= CALIBRATION_OPEN_EAR_MIN:
                if self.baseline_start is None:
                    self.baseline_start = timestamp_sec
                self.baseline_samples.append(ear)
        else:
            # Adaptation très prudente : une mesure n'est ajoutée qu'après
            # confirmation d'yeux ouverts, avec une marge au-dessus du seuil,
            # et au maximum une fois par seconde.
            old_threshold = self.threshold()
            baseline_median = (
                float(np.median(self.baseline_samples))
                if self.baseline_samples else None
            )
            plausibly_open = ear >= max(
                CALIBRATION_OPEN_EAR_MIN, old_threshold * 1.25
            )
            not_an_outlier = (
                baseline_median is None
                or ear <= baseline_median * 1.35
            )
            if (
                (self.last_eye_state == "OPEN" or not self.calibration_ready())
                and plausibly_open
                and not_an_outlier
                and (self.last_baseline_update is None
                     or timestamp_sec - self.last_baseline_update >= 1.0)
            ):
                # Si la calibration initiale était incertaine, ces mesures
                # peuvent permettre de la récupérer pendant la session.
                self.baseline_samples.append(ear)
                self.last_baseline_update = timestamp_sec

        self._apply_fallback_baseline_if_needed()
        self.last_threshold = self.threshold()

        # -------- Phase de calibration --------
        if self.in_calibration_phase():
            self.last_status = "CALIBRATING"
            self.last_eye_state = "UNKNOWN"
            self._reset_temporal_counters()
            self.samples.append({
                "t": timestamp_sec, "dt": dt,
                "valid": True, "prolonged": False,
            })
            self._prune_samples(timestamp_sec)
            self.last_perclos = self.perclos()
            self._draw_overlay(
                frame, width, height,
                status="CALIBRATING (KEEP EYES OPEN)",
                color=(255, 200, 0),
                face_box=self._face_box(landmarks, width, height),
            )
            return frame, self.snapshot()

        # -------- Classification avec confirmation temporelle --------
        threshold = self.last_threshold
        raw_closed = ear < threshold

        if raw_closed:
            if self.closed_since is None:
                self.closed_since = timestamp_sec
            self.open_since = None
        else:
            if self.open_since is None:
                self.open_since = timestamp_sec
            self.closed_since = None

        closed_confirmed = (
            self.closed_since is not None
            and timestamp_sec - self.closed_since
                >= CLOSED_CONFIRM_SECONDS
        )

        open_confirmed = (
            self.open_since is not None
            and timestamp_sec - self.open_since
                >= OPEN_CONFIRM_SECONDS
        )

        # =================================================
        # CORRECTION : création de l'événement dès que la
        # fermeture devient prolongée (et non plus à la
        # réouverture). L'événement est ensuite mis à jour
        # à chaque frame jusqu'à la réouverture.
        # =================================================
        if closed_confirmed:
            self.last_eye_state = "CLOSED"

            if self.closed_start is None:
                self.closed_start = self.closed_since

            closure_duration = timestamp_sec - self.closed_start

            if closure_duration >= MIN_CLOSURE_SECONDS:
                status = "SLEEP ALERT"

                # Chercher un événement déjà ouvert pour
                # cette fermeture
                existing = None
                for ev in self.events:
                    if (
                        ev.get("_open") is True
                        and ev["Début (s)"]
                        == round(self.closed_start, 2)
                    ):
                        existing = ev
                        break

                if existing is None:
                    self.events.append({
                        "Début (s)": round(
                            self.closed_start, 2
                        ),
                        "Fin (s)": round(timestamp_sec, 2),
                        "Durée (s)": round(
                            closure_duration, 2
                        ),
                        "Type": "Fermeture prolongée",
                        "_open": True,
                    })
                else:
                    existing["Fin (s)"] = round(
                        timestamp_sec, 2
                    )
                    existing["Durée (s)"] = round(
                        closure_duration, 2
                    )
            else:
                status = "EYES CLOSING"

        elif open_confirmed:
            self.last_eye_state = "OPEN"

            if self.closed_start is not None:
                # Finaliser l'événement ouvert
                for ev in self.events:
                    if (
                        ev.get("_open") is True
                        and ev["Début (s)"]
                        == round(self.closed_start, 2)
                    ):
                        ev["Fin (s)"] = round(
                            timestamp_sec, 2
                        )
                        ev["Durée (s)"] = round(
                            timestamp_sec
                            - self.closed_start, 2
                        )
                        ev["_open"] = False

            self.closed_start = None
            status = "AWAKE"

        else:
            status = self.last_status

        self.samples.append({
            "t": timestamp_sec,
            "dt": dt,
            "valid": True,
            "prolonged": False,
        })

        if (
            self.last_eye_state == "CLOSED"
            and self.closed_start is not None
            and timestamp_sec - self.closed_start
                >= MIN_CLOSURE_SECONDS
        ):
            for sample in self.samples:
                if sample["t"] >= self.closed_start:
                    sample["prolonged"] = True

        self._prune_samples(timestamp_sec)
        perclos = self.perclos()
        self.last_perclos = perclos

        # -------- Affichage --------
        if status == "AWAKE":
            color = (0, 200, 0)
        elif status == "SLEEP ALERT":
            color = (0, 0, 255)
        else:
            color = (0, 165, 255)

        self._draw_overlay(
            frame, width, height,
            status=status,
            color=color,
            face_box=self._face_box(landmarks, width, height),
        )

        self.last_status = status
        return frame, self.snapshot()

    # -----------------------------------------------------
    def _face_box(self, landmarks, width, height):
        xs = [int(p.x * width) for p in landmarks]
        ys = [int(p.y * height) for p in landmarks]
        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(width - 1, max(xs))
        y2 = min(height - 1, max(ys))
        return (x1, y1, x2, y2)

    def _draw_overlay(self, frame, width, height,
                      status, color, face_box):
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, status, (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )
        else:
            cv2.putText(
                frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        ear_text = (
            f"EAR: {self.last_ear:.3f}"
            if self.last_ear is not None else "EAR: --"
        )
        thr_text = f"Threshold: {self.last_threshold:.3f}"

        perclos_text = (
            f"{self.last_perclos:.1f}%"
            if self.last_perclos is not None
            else "n/a"
        )

        cv2.putText(
            frame,
            f"{ear_text} | {thr_text}",
            (20, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )
        _, valid_seconds, coverage = self.perclos_details()
        cv2.putText(
            frame,
            f"PERCLOS {PERCLOS_WINDOW_SECONDS}s: {perclos_text} "
            f"| couverture: {coverage:.0f}%",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2
        )

    # -----------------------------------------------------
    def _prune_samples(self, now):
        cutoff = now - PERCLOS_WINDOW_SECONDS
        while self.samples and self.samples[0]["t"] < cutoff:
            self.samples.popleft()

    def perclos_details(self):
        """Retourne PERCLOS et couverture de la fenêtre glissante.

        PERCLOS n'est publié qu'après une fenêtre complète de 60 s et si
        au moins 75 % de cette fenêtre contient des mesures valides.
        """
        if not self.samples or self.last_time is None:
            return None, 0.0, 0.0

        window_start = self.last_time - PERCLOS_WINDOW_SECONDS
        window_samples = [s for s in self.samples if s["t"] >= window_start]
        valid_duration = sum(s["dt"] for s in window_samples if s["valid"])
        closed_duration = sum(
            s["dt"] for s in window_samples
            if s["valid"] and s["prolonged"]
        )
        elapsed = self.last_time - (self.session_start if self.session_start is not None else self.last_time)
        coverage = min(100.0, 100.0 * valid_duration / PERCLOS_WINDOW_SECONDS)

        if elapsed < PERCLOS_WINDOW_SECONDS:
            return None, valid_duration, coverage
        if valid_duration < PERCLOS_WINDOW_SECONDS * PERCLOS_MIN_VALID_COVERAGE:
            return None, valid_duration, coverage

        return 100.0 * closed_duration / valid_duration, valid_duration, coverage

    def perclos(self):
        value, _, _ = self.perclos_details()
        return value

    def snapshot(self):
        # On expose le nombre d'événements "publics"
        public_count = len(_public_events(self.events))
        perclos_value, perclos_valid_seconds, perclos_coverage = self.perclos_details()
        return {
            "EAR": self.last_ear,
            "EAR gauche": self.last_left_ear,
            "EAR droit": self.last_right_ear,
            "Seuil": self.last_threshold,
            "PERCLOS": perclos_value,
            "Couverture PERCLOS (%)": round(perclos_coverage, 1),
            "Mesures valides PERCLOS (s)": round(perclos_valid_seconds, 1),
            "Calibration": ("Suffisante" if self.calibration_ready() else
                            ("Incertaine" if self.calibration_uncertain else "En cours")),
            "État": self.last_status,
            "Etat des yeux": self.last_eye_state,
            "Calibration suffisante": self.calibration_ready(),
            "Fermetures prolongées": public_count,
            "Fermetures prolongées (texte)": format_closure_count(
                public_count
            ),
        }

    def close(self):
        self.landmarker.close()


# =========================================================
# ANALYSE VIDEO ENREGISTRÉE
# =========================================================

def analyze_video(video_bytes, extension, preview_placeholder):
    input_path = None
    output_path = None
    h264_path = None
    detector = None
    capture = None
    writer = None

    rows = []

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension, delete=False
        ) as temp:
            temp.write(video_bytes)
            input_path = temp.name

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("Impossible d'ouvrir cette vidéo.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if src_w <= 0 or src_h <= 0:
            raise ValueError("Dimensions de vidéo invalides.")

        if src_w > MAX_OUTPUT_WIDTH:
            scale = MAX_OUTPUT_WIDTH / src_w
            width = MAX_OUTPUT_WIDTH
            height = int(src_h * scale)
        else:
            width = src_w
            height = src_h

        output_path = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ).name

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Impossible de créer la vidéo annotée.")

        detector = DrowsinessDetector(fps=fps)
        frame_id = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if (width, height) != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, (width, height))

            timestamp = frame_id / fps
            annotated, metrics = detector.process(frame, timestamp)
            writer.write(annotated)

            rows.append({
                "Temps (s)": round(timestamp, 3),
                "EAR gauche": metrics["EAR gauche"],
                "EAR droit": metrics["EAR droit"],
                "EAR": metrics["EAR"],
                "Seuil": metrics["Seuil"],
                "PERCLOS": metrics["PERCLOS"],
                "Couverture PERCLOS (%)": metrics["Couverture PERCLOS (%)"],
                "Mesures valides PERCLOS (s)": metrics["Mesures valides PERCLOS (s)"],
                "Calibration": metrics["Calibration"],
                "Etat des yeux": metrics["Etat des yeux"],
                "État": metrics["État"],
                "Fermetures prolongées": metrics[
                    "Fermetures prolongées"
                ],
                "Fermetures prolongées (texte)": metrics[
                    "Fermetures prolongées (texte)"
                ],
            })

            if frame_id % max(1, int(fps / 5)) == 0:
                preview = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )
                preview_placeholder.image(
                    preview, channels="RGB", width=PREVIEW_WIDTH
                )

            frame_id += 1

        capture.release()
        capture = None
        writer.release()
        writer = None

        # Correction 2 : fermeture encore active à la fin
        if (
            detector.closed_start is not None
            and detector.last_eye_state == "CLOSED"
        ):
            duration = (
                max(frame_id - 1, 0) / fps
                - detector.closed_start
            )
            if duration >= MIN_CLOSURE_SECONDS:
                # Finaliser l'événement déjà ouvert
                finalized = False
                for ev in detector.events:
                    if (
                        ev.get("_open") is True
                        and ev["Début (s)"]
                        == round(detector.closed_start, 2)
                    ):
                        ev["Fin (s)"] = round(
                            max(frame_id - 1, 0) / fps, 2
                        )
                        ev["Durée (s)"] = round(duration, 2)
                        ev["_open"] = False
                        finalized = True
                        break

                if not finalized:
                    detector.events.append({
                        "Début (s)": round(
                            detector.closed_start, 2
                        ),
                        "Fin (s)": round(
                            max(frame_id - 1, 0) / fps, 2
                        ),
                        "Durée (s)": round(duration, 2),
                        "Type": (
                            "Fermeture jusqu'à la fin de la vidéo"
                        ),
                        "_open": False,
                    })

        events = _public_events(detector.events)
        detector.close()
        detector = None

        if frame_id == 0:
            raise ValueError("Aucune image lisible dans la vidéo.")

        h264_path = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ).name

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg, "-y", "-i", output_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an", h264_path,
            ],
            check=True,
            capture_output=True,
        )

        with open(h264_path, "rb") as f:
            annotated_bytes = f.read()

        return (
            pd.DataFrame(rows),
            pd.DataFrame(events),
            annotated_bytes,
        )

    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if detector is not None:
            detector.close()

        for path in (input_path, output_path, h264_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# =========================================================
# WEBCAM EN DIRECT
# =========================================================

class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = DrowsinessDetector()
        self.start_time = time.monotonic()
        self.latest_snapshot = self.detector.snapshot()
        self.rows = []

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        if image is None or image.size == 0:
            return frame

        h, w = image.shape[:2]
        if w > 640:
            scale = 640 / w
            image = cv2.resize(
                image, (640, int(h * scale))
            )

        timestamp = time.monotonic() - self.start_time
        annotated, metrics = self.detector.process(
            image, timestamp
        )

        self.latest_snapshot = metrics
        self.rows.append({
            "Temps (s)": round(timestamp, 3),
            "EAR gauche": metrics["EAR gauche"],
            "EAR droit": metrics["EAR droit"],
            "EAR": metrics["EAR"],
            "Seuil": metrics["Seuil"],
            "PERCLOS": metrics["PERCLOS"],
            "Couverture PERCLOS (%)": metrics["Couverture PERCLOS (%)"],
            "Mesures valides PERCLOS (s)": metrics["Mesures valides PERCLOS (s)"],
            "Calibration": metrics["Calibration"],
            "Etat des yeux": metrics["Etat des yeux"],
            "État": metrics["État"],
            "Fermetures prolongées": metrics[
                "Fermetures prolongées"
            ],
            "Fermetures prolongées (texte)": metrics[
                "Fermetures prolongées (texte)"
            ],
        })

        if len(self.rows) > 100_000:
            self.rows = self.rows[-50_000:]

        return av.VideoFrame.from_ndarray(
            annotated, format="bgr24"
        )

    def get_rows(self):
        return list(self.rows)

    def get_events(self):
        events = _public_events(self.detector.events)

        # Correction 3 : fermeture en cours au moment de l'arrêt
        if (
            self.detector.closed_start is not None
            and self.detector.last_eye_state == "CLOSED"
            and self.detector.last_time is not None
        ):
            duration = (
                self.detector.last_time
                - self.detector.closed_start
            )
            if duration >= MIN_CLOSURE_SECONDS:
                # Finaliser l'événement déjà ouvert s'il existe
                finalized = False
                for ev in events:
                    if ev["Début (s)"] == round(
                        self.detector.closed_start, 2
                    ):
                        ev["Fin (s)"] = round(
                            self.detector.last_time, 2
                        )
                        ev["Durée (s)"] = round(duration, 2)
                        finalized = True
                        break

                if not finalized:
                    events.append({
                        "Début (s)": round(
                            self.detector.closed_start, 2
                        ),
                        "Fin (s)": round(
                            self.detector.last_time, 2
                        ),
                        "Durée (s)": round(duration, 2),
                        "Type": (
                            "Fermeture à l'arrêt de la webcam"
                        ),
                    })

        return events


# =========================================================
# INTERFACE STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Driver Drowsiness Detection")
st.caption(
    "EAR personnalisé • PERCLOS glissant sur 60 s avec couverture "
    "affichée • Analyse vidéo et webcam"
)
st.warning(
    "Prototype académique uniquement. Ne l'utilise pas comme dispositif "
    "de sécurité pendant la conduite réelle. Le seuil PERCLOS est expérimental."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Le fichier face_landmarker.task est absent. "
        "Place-le dans le même dossier que app.py."
    )
    st.stop()

mode = st.radio(
    "Choisir le mode",
    ["Vidéo enregistrée", "Webcam en direct"],
    horizontal=True,
)


# ---------------------------------------------------------
# MODE VIDÉO ENREGISTRÉE
# ---------------------------------------------------------
if mode == "Vidéo enregistrée":
    uploaded = st.file_uploader(
        "Importer une vidéo du conducteur",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded and st.button(
        "Analyser la vidéo", type="primary"
    ):
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("Aperçu pendant l'analyse")
            preview = st.empty()

        with right:
            st.subheader("Traitement")
            status = st.empty()
            status.info("Analyse en cours...")

        try:
            with st.spinner("Traitement des images..."):
                df, events_df, annotated_bytes = analyze_video(
                    uploaded.getvalue(),
                    os.path.splitext(uploaded.name)[1].lower(),
                    preview,
                )

            status.success("Analyse terminée.")
            st.subheader("Vidéo annotée")
            st.video(annotated_bytes, width=400)

            if not df.empty:
                latest = df.iloc[-1]

                # Correction 4 : diagnostic calibration
                seuil_final = latest["Seuil"]
                if latest.get("Calibration", "Incertaine") != "Suffisante":
                    st.warning(
                        "Calibration incertaine : nombre insuffisant de mesures "
                        "plausibles d'yeux ouverts. Le détecteur utilise le seuil "
                        f"de secours fixe ({INITIAL_THRESHOLD:.2f}). Vérifiez la "
                        "vidéo et la qualité de détection."
                    )

                k1, k2, k3, k4, k5 = st.columns(5)

                k1.metric(
                    "EAR final",
                    (
                        f"{latest['EAR']:.3f}"
                        if pd.notna(latest["EAR"])
                        else "Indisponible"
                    ),
                )
                k2.metric(
                    "Seuil final",
                    (
                        f"{latest['Seuil']:.3f}"
                        if pd.notna(latest["Seuil"])
                        else "Indisponible"
                    ),
                )
                k3.metric(
                    "PERCLOS",
                    (
                        f"{latest['PERCLOS']:.1f}%"
                        if pd.notna(latest["PERCLOS"])
                        else "n/a"
                    ),
                )
                k4.metric(
                    "Fermetures prolongées",
                    str(
                        latest["Fermetures prolongées (texte)"]
                    ),
                )
                k5.metric(
                    "État final",
                    str(latest["État"]),
                )
                st.caption(
                    f"Calibration : {latest.get('Calibration', 'Inconnue')} · "
                    f"Couverture PERCLOS : "
                    f"{latest.get('Couverture PERCLOS (%)', 0):.1f}% "
                    f"({latest.get('Mesures valides PERCLOS (s)', 0):.1f} s valides)"
                )

                if (
                    pd.notna(latest["PERCLOS"])
                    and latest["PERCLOS"] >= PERCLOS_ALERT
                ):
                    st.warning(
                        f"PERCLOS ≥ {PERCLOS_ALERT}% : "
                        "indice expérimental de somnolence élevé (à valider sur des données annotées)."
                    )

                st.subheader("Évolution de l'EAR")
                chart_df = df.set_index("Temps (s)")
                st.line_chart(
                    chart_df[["EAR", "Seuil"]].dropna()
                )

                # ===== Tableau 1 : événements =====
                st.subheader("Journal des fermetures prolongées")
                if events_df.empty:
                    st.info(
                        "Aucune fermeture prolongée détectée."
                    )
                else:
                    st.dataframe(
                        events_df, use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Télécharger les événements (CSV)",
                        data=events_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="drowsiness_events.csv",
                        mime="text/csv",
                    )

                # ===== Tableau 2 : image par image =====
                st.subheader("Données image par image")
                display_df = df.drop(
                    columns=["Fermetures prolongées"],
                    errors="ignore",
                )
                st.dataframe(
                    display_df, use_container_width=True,
                    hide_index=True,
                )

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger les mesures CSV",
                    data=csv,
                    file_name="drowsiness_results.csv",
                    mime="text/csv",
                )

        except Exception as error:
            status.error(f"Erreur : {error}")


# ---------------------------------------------------------
# MODE WEBCAM
# ---------------------------------------------------------
else:
    st.info(
        "Autorise l'accès à la caméra dans ton navigateur. "
        "Le traitement commence lorsque la connexion vidéo démarre."
    )

    ctx = webrtc_streamer(
        key="drowsiness-webcam",
        video_processor_factory=LiveProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st_autorefresh(interval=1000, key="webcam_refresh")

    st.caption(
        "Les mesures EAR, le seuil, le PERCLOS et l'état sont "
        "affichés sur l'image et dans le panneau à droite. "
        "La calibration est progressive."
    )

    # -----------------------------------------------------
    # Panneau temps réel
    # -----------------------------------------------------
    if ctx.state.playing and ctx.video_processor:
        snapshot = ctx.video_processor.latest_snapshot

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "EAR",
            (
                f"{snapshot['EAR']:.3f}"
                if snapshot["EAR"] is not None
                else "--"
            ),
        )
        m2.metric(
            "Seuil",
            f"{snapshot['Seuil']:.3f}",
        )
        m3.metric(
            "PERCLOS",
            (
                f"{snapshot['PERCLOS']:.1f}%"
                if snapshot["PERCLOS"] is not None
                else "n/a"
            ),
        )
        m4.metric(
            "État",
            str(snapshot["État"]),
        )

        if snapshot["Calibration"] == "Suffisante":
            st.success("Calibration suffisante : baseline calculée à partir d'yeux plausiblement ouverts.")
        elif snapshot["Calibration"] == "Incertaine":
            st.warning(
                f"Calibration incertaine : seuil fixe {INITIAL_THRESHOLD:.2f} utilisé. "
                "Regarde la caméra avec les yeux ouverts et un visage bien éclairé."
            )
        else:
            st.info("Calibration en cours : garde le visage face à la caméra et les yeux ouverts.")
        st.caption(
            f"Couverture PERCLOS : {snapshot['Couverture PERCLOS (%)']:.1f}% "
            f"({snapshot['Mesures valides PERCLOS (s)']:.1f} s valides sur 60 s). "
            "Le PERCLOS reste indisponible tant que la fenêtre n'est pas suffisamment complète."
        )

        live_rows = ctx.video_processor.get_rows()
        if live_rows:
            df_live_rt = pd.DataFrame(live_rows)
            df_live_rt = df_live_rt.tail(300)
            df_live_rt = df_live_rt.set_index("Temps (s)")
            st.subheader("EAR en temps réel")
            st.line_chart(
                df_live_rt[["EAR", "Seuil"]].dropna()
            )

    # -----------------------------------------------------
    # Analyse après fin de webcam
    # -----------------------------------------------------
    st.divider()
    st.subheader("Analyse de la session webcam")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Analyser la session webcam"):
            if ctx.video_processor is None:
                st.warning(
                    "Aucune session webcam active ou "
                    "processeur indisponible."
                )
            else:
                rows = ctx.video_processor.get_rows()
                events = ctx.video_processor.get_events()

                if not rows:
                    st.warning(
                        "Aucune donnée collectée pour le moment."
                    )
                else:
                    st.session_state["webcam_rows"] = rows
                    st.session_state["webcam_events"] = events
                    st.success(
                        f"{len(rows)} échantillons et "
                        f"{len(events)} fermetures prolongées."
                    )

    with col_b:
        if st.button("Réinitialiser la session"):
            st.session_state.pop("webcam_rows", None)
            st.session_state.pop("webcam_events", None)
            st.info("Session réinitialisée.")

    if "webcam_rows" in st.session_state:
        rows = st.session_state["webcam_rows"]
        events = st.session_state["webcam_events"]

        df_live = pd.DataFrame(rows)
        events_df_live = pd.DataFrame(events)

        if not df_live.empty:
            latest = df_live.iloc[-1]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "EAR final",
                (
                    f"{latest['EAR']:.3f}"
                    if pd.notna(latest["EAR"])
                    else "--"
                ),
            )
            k2.metric(
                "Seuil final",
                (
                    f"{latest['Seuil']:.3f}"
                    if pd.notna(latest["Seuil"])
                    else "--"
                ),
            )
            k3.metric(
                "PERCLOS",
                (
                    f"{latest['PERCLOS']:.1f}%"
                    if pd.notna(latest["PERCLOS"])
                    else "n/a"
                ),
            )
            k4.metric(
                "Fermetures prolongées",
                str(latest["Fermetures prolongées (texte)"]),
            )
            st.caption(
                f"Calibration : {latest.get('Calibration', 'Inconnue')} · "
                f"Couverture PERCLOS : {latest.get('Couverture PERCLOS (%)', 0):.1f}% "
                f"({latest.get('Mesures valides PERCLOS (s)', 0):.1f} s valides)"
            )

            if (
                pd.notna(latest["PERCLOS"])
                and latest["PERCLOS"] >= PERCLOS_ALERT
            ):
                st.warning(
                    f"PERCLOS ≥ {PERCLOS_ALERT}% : "
                    "indice expérimental de somnolence élevé (à valider sur des données annotées)."
                )

            st.subheader("Évolution de l'EAR (webcam)")
            chart_df = df_live.set_index("Temps (s)")
            st.line_chart(
                chart_df[["EAR", "Seuil"]].dropna()
            )

            st.subheader(
                "Journal des fermetures prolongées (webcam)"
            )
            if events_df_live.empty:
                st.info(
                    "Aucune fermeture prolongée détectée."
                )
            else:
                st.dataframe(
                    events_df_live,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Télécharger les événements webcam (CSV)",
                    data=events_df_live.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="webcam_events.csv",
                    mime="text/csv",
                )

            st.subheader(
                "Données image par image (webcam)"
            )
            display_df_live = df_live.drop(
                columns=["Fermetures prolongées"],
                errors="ignore",
            )
            st.dataframe(
                display_df_live,
                use_container_width=True,
                hide_index=True,
            )

            csv = df_live.to_csv(
                index=False
            ).encode("utf-8")
            st.download_button(
                "Télécharger les mesures webcam (CSV)",
                data=csv,
                file_name="webcam_measures.csv",
                mime="text/csv",
            )


st.divider()
st.caption(
    "Prototype académique : ce système n'est pas certifié "
    "pour la conduite réelle et ne remplace pas les dispositifs "
    "de sécurité automobile."
)