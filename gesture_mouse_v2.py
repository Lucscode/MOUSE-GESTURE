"""
Gesture Mouse v2 — Controle o mouse com gestos da mão via webcam.

Gestos:
  - Indicador apontando      → mover cursor
  - Pinça (polegar+indicador) → clique esquerdo
  - Pinça (polegar+médio)     → clique direito
  - Indicador+médio levantados, demais fechados → modo SCROLL
      (mova a mão pra cima/baixo pra rolar)
  - Pinça segurada >0.4s      → arrastar (drag); soltar pinça = drop
  - Joia (👍) por 2s          → pausar / retomar
  - Q na janela               → sair

Roda na system tray ao minimizar a janela.
"""

import cv2
import time
import math
import numpy as np
import mediapipe as mp
import pyautogui
import ctypes
import ctypes.wintypes
import os
import sys
import urllib.request
import threading

# ===== Configs =====
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 60   # FPS solicitado à câmera (real depende do hardware)

# --- Suavização (One Euro Filter) ---
OEF_MIN_CUTOFF = 2.5       # ↓ = mais suave.  ↑ = menos filtro
OEF_BETA       = 0.7       # ↑ = mais responsivo em movimentos rápidos
OEF_D_CUTOFF   = 1.0

# --- Clique ---
PINCH_THRESHOLD     = 0.06   # polegar↔indicador pra ENTRAR na pinça
PINCH_RELEASE       = 0.12   # polegar↔indicador pra SAIR da pinça (histerese ampla)
CLICK_COOLDOWN      = 0.45   # segundos entre cliques
HOLD_DELAY          = 0.12   # segundos: pinça < isso = clique, >= isso = manter pressionado

# --- Scroll ---
SCROLL_SPEED        = 8      # linhas por frame de deslocamento
SCROLL_DEAD_ZONE    = 0.015  # zona morta vertical (normalizada)

# --- Mapeamento câmera → tela ---
MARGIN_X = 0.28     # margem horizontal (0.28 = usa 44% central do frame)
MARGIN_TOP = 0.15   # margem do topo (menos — câmera fica acima da mão)
MARGIN_BOT = 0.05   # margem de baixo (bem pouco — facilita chegar na taskbar)

# --- Anti-tremor ---
DEAD_ZONE_PX = 0.5

# --- Joia (pause/resume) ---
THUMBS_HOLD_TIME = 2.0
THUMBS_COOLDOWN  = 1.5

# --- Sistema ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()


# ============================================================
#  ONE EURO FILTER
# ============================================================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
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
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


filter_x = OneEuroFilter(OEF_MIN_CUTOFF, OEF_BETA, OEF_D_CUTOFF)
filter_y = OneEuroFilter(OEF_MIN_CUTOFF, OEF_BETA, OEF_D_CUTOFF)


# ============================================================
#  WIN32 CURSOR (zero latência)
# ============================================================
user32 = ctypes.windll.user32

# mouse_event flags
MOUSEEVENTF_MOVE       = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_ABSOLUTE   = 0x8000

def move_cursor(x, y):
    """Move o cursor via mouse_event com flag ABSOLUTE.
    Isso gera WM_MOUSEMOVE real — necessário para apps como Paint."""
    # Coordenadas absolutas: 0-65535 mapeadas à tela
    abs_x = int(x * 65536 / screen_w)
    abs_y = int(y * 65536 / screen_h)
    user32.mouse_event(
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
        abs_x, abs_y, 0, 0
    )

def mouse_down(button="left"):
    flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
    user32.mouse_event(flag, 0, 0, 0, 0)

def mouse_up(button="left"):
    flag = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP
    user32.mouse_event(flag, 0, 0, 0, 0)


# ============================================================
#  HELPERS DE GESTO
# ============================================================
def clamp(v, a, b):
    return max(a, min(b, v))

def remap(v, lo, hi):
    return clamp((v - lo) / (hi - lo), 0.0, 1.0)

def finger_extended(lms, tip, pip):
    """Dedo estendido se ponta (tip) estiver acima (y menor) do PIP."""
    return lms[tip].y < lms[pip].y

def thumb_extended(lms):
    """Polegar estendido se ponta (4) mais afastada horizontalmente do wrist que o IP (3)."""
    return abs(lms[4].x - lms[0].x) > abs(lms[3].x - lms[0].x)

def detect_gesture(lms):
    """Retorna o gesto atual baseado nos landmarks.
    Possíveis: 'scroll', 'move'
    """
    middle_up = finger_extended(lms, 12, 10)
    ring_up   = finger_extended(lms, 16, 14)
    pinky_up  = finger_extended(lms, 20, 18)

    # Scroll: dedo do meio estendido, anelar + mindinho fechados
    if middle_up and not ring_up and not pinky_up:
        return "scroll"

    return "move"


# ============================================================
#  MEDIAPIPE SETUP
# ============================================================
USE_SOLUTIONS = hasattr(mp, "solutions")

if USE_SOLUTIONS:
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils
else:
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker, HandLandmarkerOptions,
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hand_landmarker.task")

    if not os.path.exists(MODEL_PATH):
        print("[mediapipe] Baixando modelo hand_landmarker.task …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as e:
            raise RuntimeError(
                "Falhou ao baixar o modelo. Baixe manualmente e salve como "
                "'hand_landmarker.task' na pasta do script."
            ) from e

    _opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = HandLandmarker.create_from_options(_opts)
    _t0 = time.time()


# ============================================================
#  SYSTEM TRAY (roda em thread separada)
# ============================================================
tray_icon = None

def _build_tray():
    """Cria ícone na bandeja do sistema."""
    try:
        import pystray
        from PIL import Image as PilImage, ImageDraw
    except ImportError:
        return  # sem pystray, ignora

    # ícone simples: quadrado verde com um cursor branco
    img = PilImage.new("RGB", (64, 64), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.ellipse([16, 16, 48, 48], fill=(0, 200, 100))
    d.polygon([(28, 20), (28, 44), (40, 36)], fill="white")

    def on_quit(icon, item):
        global _quit_flag
        _quit_flag = True
        icon.stop()

    def on_show(icon, item):
        global _show_window
        _show_window = True

    menu = pystray.Menu(
        pystray.MenuItem("Mostrar janela", on_show, default=True),
        pystray.MenuItem("Sair", on_quit),
    )
    global tray_icon
    tray_icon = pystray.Icon("GestureMouse", img, "Gesture Mouse", menu)
    tray_icon.run()


_quit_flag = False
_show_window = False
tray_thread = threading.Thread(target=_build_tray, daemon=True)
tray_thread.start()


# ============================================================
#  MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # DirectShow = menos latência no Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # buffer mínimo = frame mais recente

prev_x, prev_y = screen_w / 2, screen_h / 2
last_click_t = 0.0
fps_t = time.time()
fps = 0
fps_count = 0

# Estado

# Drag
dragging = False
pinch_start_t = None   # quando a pinça começou
pinch_active = False   # True enquanto estiver em estado de pinça (histerese)

# Scroll
scroll_anchor_y = None  # y normalizado quando entrou em modo scroll

# Janela
window_name = "Gesture Mouse v2"
window_hidden = False


def draw_text(img, text, pos, scale=0.7, color=(255, 255, 255),
              thickness=1, shadow=True):
    if shadow:
        cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness+2)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(frame, status, pinch_dist, gesture, fps_val):
    """Desenha o HUD com informações de estado."""
    h, w = frame.shape[:2]

    # Faixa semitransparente no topo
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Status principal
    status_colors = {
        "MOVE": (0, 255, 100),
        "CLICK!": (0, 200, 255),
        "PINCH": (0, 180, 255),
        "HOLD": (255, 165, 0),
        "SCROLL": (255, 255, 0),
        "NO HAND": (150, 150, 150),
    }
    col = status_colors.get(status, (255, 255, 255))
    draw_text(frame, f"Status: {status}", (10, 30), 0.85, col, 2)
    draw_text(frame, f"Gesto: {gesture}", (10, 60), 0.55, (200, 200, 200), 1)

    # FPS
    draw_text(frame, f"FPS: {fps_val}", (w - 120, 30), 0.6, (180, 180, 180), 1)

    # Pinch distance
    if pinch_dist is not None:
        bar_max = 0.15
        bar_pct = clamp(pinch_dist / bar_max, 0, 1)
        bx, by, bw, bh = 10, 75, 180, 8
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
        bar_col = (0, 255, 0) if pinch_dist > PINCH_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw * bar_pct), by + bh), bar_col, -1)

    # Legenda de gestos (canto inferior)
    legend = [
        "Pulso: Mover cursor",
        "Pinca (pol+ind): Soltar=Clique | Manter=Segurar",
        "Dedo medio: Scroll cima/baixo",
        "Q: Sair | M: Minimizar",
    ]
    y0 = h - 15 * len(legend) - 5
    for i, line in enumerate(legend):
        draw_text(frame, line, (10, y0 + i * 15), 0.38,
                  (180, 180, 180), 1, shadow=False)


while not _quit_flag:
    ok, frame = cap.read()
    if not ok:
        break

    # --- Mostrar/Esconder janela via tray ---
    if _show_window and window_hidden:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        window_hidden = False
        _show_window = False

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if USE_SOLUTIONS:
        res = hands_detector.process(rgb)
    else:
        timestamp_ms = int((time.time() - _t0) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_image, timestamp_ms)

    status = "NO HAND"
    pinch_dist = None
    rclick_dist = None
    gesture = "—"

    hand_detected = ((USE_SOLUTIONS and res.multi_hand_landmarks)
                     or (not USE_SOLUTIONS and res.hand_landmarks))

    if hand_detected:
        if USE_SOLUTIONS:
            lm_raw = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm_raw, mp_hands.HAND_CONNECTIONS)
            landmarks = lm_raw.landmark
        else:
            landmarks = res.hand_landmarks[0]

        gesture = detect_gesture(landmarks)
        now = time.time()

        # Coordenadas-chave
        px, py = landmarks[9].x,  landmarks[9].y   # palma (MCP médio) — cursor
        ix, iy = landmarks[8].x,  landmarks[8].y   # ponta indicador
        tx, ty = landmarks[4].x,  landmarks[4].y   # ponta polegar
        mx, my = landmarks[12].x, landmarks[12].y  # ponta médio

        pinch_dist = float(np.hypot(ix - tx, iy - ty))

        # ===== SCROLL MODE (dedo do meio estendido) =====
        if gesture == "scroll":
            # Âncora: posição y do médio quando entrou no modo
            if scroll_anchor_y is None:
                scroll_anchor_y = my
            delta = my - scroll_anchor_y
            if abs(delta) > SCROLL_DEAD_ZONE:
                lines = int(delta / SCROLL_DEAD_ZONE) * SCROLL_SPEED
                pyautogui.scroll(-lines)
                scroll_anchor_y = my
            status = "SCROLL"

            # Desenhar indicador de scroll
            h_f, w_f = frame.shape[:2]
            cy = int(my * h_f)
            cx = int(mx * w_f)
            cv2.arrowedLine(frame, (cx, cy), (cx, cy - 30),
                            (255, 255, 0), 3, tipLength=0.4)
            cv2.arrowedLine(frame, (cx, cy), (cx, cy + 30),
                            (255, 255, 0), 3, tipLength=0.4)

        else:
            scroll_anchor_y = None

            # --- Histerese de pinça (polegar + indicador) ---
            if not pinch_active and pinch_dist < PINCH_THRESHOLD:
                pinch_active = True
                pinch_start_t = now

            # --- Mover cursor pela PALMA (sempre, inclusive durante drag) ---
            norm_x = remap(px, MARGIN_X, 1.0 - MARGIN_X)
            norm_y = remap(py, MARGIN_TOP, 1.0 - MARGIN_BOT)
            target_x = norm_x * (screen_w - 1)
            target_y = norm_y * (screen_h - 1)

            now_t = time.perf_counter()
            new_x = filter_x(target_x, now_t)
            new_y = filter_y(target_y, now_t)

            if math.hypot(new_x - prev_x, new_y - prev_y) > DEAD_ZONE_PX:
                move_cursor(new_x, new_y)
                prev_x, prev_y = new_x, new_y

            if not pinch_active:
                status = "MOVE"

            # ===== CLIQUE ESQ / HOLD (pinça polegar + indicador) =====
            if pinch_active:
                if pinch_dist > PINCH_RELEASE:
                    # Soltou a pinça
                    pinch_active = False
                    held = now - pinch_start_t if pinch_start_t else 0
                    if dragging:
                        mouse_up("left")
                        dragging = False
                        status = "MOVE"
                    elif held < HOLD_DELAY and now - last_click_t > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click_t = now
                        status = "CLICK!"
                    else:
                        status = "MOVE"
                    pinch_start_t = None
                else:
                    held = now - pinch_start_t if pinch_start_t else 0
                    if not dragging and held >= HOLD_DELAY:
                        mouse_down("left")
                        dragging = True
                    status = "HOLD" if dragging else "PINCH"

        # UI: marcar pontos
        h_f, w_f = frame.shape[:2]
        cv2.circle(frame, (int(px * w_f), int(py * h_f)),
                   10, (0, 255, 255), -1)  # palma (cursor)
        cv2.circle(frame, (int(ix * w_f), int(iy * h_f)),
                   6, (0, 255, 0), -1)     # indicador
        cv2.circle(frame, (int(tx * w_f), int(ty * h_f)),
                   6, (255, 0, 0), -1)     # polegar
        cv2.circle(frame, (int(mx * w_f), int(my * h_f)),
                   6, (255, 0, 255), -1)   # médio

    else:
        # Sem mão → soltar drag se ativo
        scroll_anchor_y = None
        if dragging:
            mouse_up("left")
            dragging = False
            pinch_start_t = None

    # FPS
    fps_count += 1
    dt = time.time() - fps_t
    if dt >= 0.5:
        fps = int(fps_count / dt)
        fps_count = 0
        fps_t = time.time()

    # HUD
    draw_hud(frame, status, pinch_dist, gesture, fps)

    if not window_hidden:
        cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key == ord('m') or key == ord('M'):
        # Minimizar pra tray
        cv2.destroyWindow(window_name)
        window_hidden = True

# Cleanup
if dragging:
    mouse_up("left")

cap.release()
cv2.destroyAllWindows()

if not USE_SOLUTIONS:
    landmarker.close()

if tray_icon is not None:
    tray_icon.stop()
