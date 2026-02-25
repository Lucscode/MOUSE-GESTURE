import cv2
import time
import math
import numpy as np
import mediapipe as mp
import pyautogui
import ctypes
import os
import urllib.request

# ===== Configs =====
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# One Euro Filter — ajuste fino de suavização
OEF_MIN_CUTOFF = 3.0      # cutoff baixo = mais suave (↓ tremor). Aumente pra menos filtro.
OEF_BETA       = 0.5      # beta alto = mais responsivo em movimentos rápidos
OEF_D_CUTOFF   = 1.0      # cutoff do derivativo (geralmente não precisa mexer)

PINCH_THRESHOLD = 0.07    # distância normalizada polegar↔indicador (maior = precisa juntar mais pra clicar)
CLICK_COOLDOWN  = 0.5     # segundos entre cliques

# Margem: ignora a faixa [0..MARGIN] e [1-MARGIN..1] da câmera,
# mapeando só a região central pra tela toda (mais confortável).
MARGIN = 0.25

# Dead zone: ignora movimentos menores que N pixels (anti-micro-tremor)
DEAD_ZONE_PX = 0.5

# Gesto de palma aberta ("pare") pausa/retoma o controle
PALM_HOLD_TIME  = 2.0     # segundos segurando a joia pra ativar
PALM_COOLDOWN   = 1.5     # segundos de espera antes de poder alternar de novo

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0        # remove o delay padrão de 0.1 s do pyautogui
screen_w, screen_h = pyautogui.size()

# ---------- One Euro Filter (adaptativo) ----------
class OneEuroFilter:
    """Filtra ruído com latência mínima. Movimentos lentos = mais suave;
    movimentos rápidos = menos atraso."""
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        if t is None:
            t = time.perf_counter()
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        # derivada filtrada
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # cutoff adaptativo
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

filter_x = OneEuroFilter(OEF_MIN_CUTOFF, OEF_BETA, OEF_D_CUTOFF)
filter_y = OneEuroFilter(OEF_MIN_CUTOFF, OEF_BETA, OEF_D_CUTOFF)

# ---------- Cursor via Win32 (zero latência) ----------
def move_cursor(x, y):
    ctypes.windll.user32.SetCursorPos(int(x), int(y))

USE_SOLUTIONS = hasattr(mp, "solutions")

if USE_SOLUTIONS:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    mp_draw = mp.solutions.drawing_utils
else:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

    if not os.path.exists(MODEL_PATH):
        print("[mediapipe] Baixando modelo hand_landmarker.task...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                "Falhou ao baixar o modelo do MediaPipe. "
                "Baixe manualmente e salve como 'hand_landmarker.task' na mesma pasta do script."
            ) from e

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = HandLandmarker.create_from_options(options)
    t0 = time.time()

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

prev_x, prev_y = screen_w / 2, screen_h / 2
last_click_t = 0.0
fps_t = time.time()
fps = 0

def clamp(v, a, b):
    return max(a, min(b, v))

def remap(v, lo, hi):
    """Mapeia v de [lo, hi] → [0, 1], clampado."""
    return clamp((v - lo) / (hi - lo), 0.0, 1.0)

def is_thumbs_up(lms):
    """Retorna True se o gesto for 'joia' (polegar pra cima, demais dedos fechados)."""
    # Polegar: ponta (4) bem acima (y menor) do MCP (2)
    thumb_up = lms[4].y < lms[3].y < lms[2].y
    # Demais dedos: ponta ABAIXO (y maior) do PIP → fechados
    index_closed  = lms[8].y  > lms[6].y
    middle_closed = lms[12].y > lms[10].y
    ring_closed   = lms[16].y > lms[14].y
    pinky_closed  = lms[20].y > lms[18].y
    return thumb_up and index_closed and middle_closed and ring_closed and pinky_closed

paused = False
thumb_start_t = None       # quando a joia começou a ser detectada
last_toggle_t = 0.0        # último momento em que o estado foi alternado

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if USE_SOLUTIONS:
        res = hands.process(rgb)
    else:
        timestamp_ms = int((time.time() - t0) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, timestamp_ms)

    status = "NO HAND" if not paused else "PAUSED"
    pinch_dist = None

    hand_detected = ((USE_SOLUTIONS and res.multi_hand_landmarks)
                     or ((not USE_SOLUTIONS) and res.hand_landmarks))

    if hand_detected:
        if USE_SOLUTIONS:
            lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            landmarks = lm.landmark
        else:
            landmarks = res.hand_landmarks[0]

        # --- Detecção de joia (toggle pause) ---
        now = time.time()
        if is_thumbs_up(landmarks):
            if thumb_start_t is None:
                thumb_start_t = now
            elif (now - thumb_start_t >= PALM_HOLD_TIME
                  and now - last_toggle_t >= PALM_COOLDOWN):
                paused = not paused
                last_toggle_t = now
                thumb_start_t = None
                # resetar filtros ao retomar pra evitar "salto" do cursor
                if not paused:
                    filter_x.__init__(OEF_MIN_CUTOFF, OEF_BETA, OEF_D_CUTOFF)
                    filter_y.__init__(OEF_MIN_CUTOFF, OEF_BETA, OEF_D_CUTOFF)
        else:
            thumb_start_t = None

        if paused or thumb_start_t is not None:
            status = "PAUSED" if paused else "JOIA..."
        else:
            # Landmarks úteis
            # 8: ponta do indicador, 4: ponta do polegar
            ix, iy = landmarks[8].x, landmarks[8].y
            tx, ty = landmarks[4].x, landmarks[4].y

            pinch_dist = float(np.hypot(ix - tx, iy - ty))

            # Remapear com margem → [0,1] e depois → pixels da tela
            norm_x = remap(ix, MARGIN, 1.0 - MARGIN)
            norm_y = remap(iy, MARGIN, 1.0 - MARGIN)
            target_x = norm_x * (screen_w - 1)
            target_y = norm_y * (screen_h - 1)

            # One Euro Filter (suavização adaptativa)
            now_t = time.perf_counter()
            new_x = filter_x(target_x, now_t)
            new_y = filter_y(target_y, now_t)

            # Dead zone: só move se deslocamento > limiar
            if math.hypot(new_x - prev_x, new_y - prev_y) > DEAD_ZONE_PX:
                move_cursor(new_x, new_y)
                prev_x, prev_y = new_x, new_y

            if pinch_dist < PINCH_THRESHOLD:
                status = "PINCH"
                if now - last_click_t > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_t = now
                    status = "CLICK!"
            else:
                status = "MOVE"

            # UI: marcar pontos
            h, w = frame.shape[:2]
            cv2.circle(frame, (int(ix * w), int(iy * h)), 10, (0, 255, 0), -1)
            cv2.circle(frame, (int(tx * w), int(ty * h)), 10, (255, 0, 0), -1)
    else:
        thumb_start_t = None

    # FPS simples
    dt = time.time() - fps_t
    if dt >= 0.5:
        fps = int(1.0 / (dt / max(1, int(dt * 30))))
        fps_t = time.time()

    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)
    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

    if pinch_dist is not None:
        cv2.putText(frame, f"Pinch: {pinch_dist:.3f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Overlay de PAUSED
    if paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        txt = "PAUSED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 2.0, 4
        (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
        cx = (frame.shape[1] - tw) // 2
        cy = (frame.shape[0] + th) // 2
        cv2.putText(frame, txt, (cx, cy), font, scale, (0, 0, 255), thick)
        cv2.putText(frame, "Joia para retomar", (cx - 10, cy + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    # Barra de progresso da joia (feedback visual)
    elif thumb_start_t is not None:
        progress = min((time.time() - thumb_start_t) / PALM_HOLD_TIME, 1.0)
        bar_w = int(frame.shape[1] * 0.6)
        bar_x = (frame.shape[1] - bar_w) // 2
        bar_y = frame.shape[0] - 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20),
                      (80, 80, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + 20),
                      (0, 140, 255), -1)
        cv2.putText(frame, "Segure joia...",
                    (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 140, 255), 1)

    cv2.putText(frame, "Pressione Q para sair", (10, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Gesture Mouse (MVP)", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()

if not USE_SOLUTIONS:
    landmarker.close()