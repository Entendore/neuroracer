#!/usr/bin/env python
"""Optional OpenGL renderer for NeuroRacer.
Used automatically by neuroracer.py when importable; set NEURORACER_GL=0 to
force the raster QPainter view. Requires numpy and a GL 3.1+ context."""
import colorsys
import math

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QVector2D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram
from PySide6.QtOpenGLWidgets import QOpenGLWidget

import neuroracer as nr
from neuroracer import CameraMixin

GL_LINES = 0x0001
GL_TRIANGLES = 0x0004
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_COLOR_BUFFER_BIT = 0x00004000
GL_FLOAT = 0x1406

_VS = """#version 130
uniform vec2 res;
uniform vec2 cam;
uniform float scale;
in vec2 pos;
in vec4 col;
out vec4 vcol;
void main() {
    vec2 s = (pos - cam) * scale + 0.5 * res;
    gl_Position = vec4(2.0 * s.x / res.x - 1.0, 1.0 - 2.0 * s.y / res.y, 0.0, 1.0);
    vcol = col;
}
"""
_FS = """#version 130
in vec4 vcol;
out vec4 frag;
void main() { frag = vec4(vcol.rgb * vcol.a, vcol.a); }
"""

ASPHALT = (53 / 255, 57 / 255, 66 / 255, 1.0)
WALL = (222 / 255, 226 / 255, 235 / 255, 1.0)
CENTER = (235 / 255, 200 / 255, 90 / 255, 0.40)
RACING = (90 / 255, 230 / 255, 140 / 255, 0.45)
GHOST_C = (1.0, 1.0, 1.0, 0.30)
DEAD = (225 / 255, 80 / 255, 80 / 255, 0.16)
PLAYER = (1.0, 0.47, 0.16, 1.0)
CAR_PTS = [(p.x(), p.y()) for p in nr.CAR_SHAPE]


def _hue_rgb(hue, s=0.60, v=0.95, a=1.0):
    r, g, b = colorsys.hsv_to_rgb((0.52 + 0.92 * hue) % 1.0, s, v)
    return (r, g, b, a)


def _fan(pts, col):
    out = np.empty((3 * (len(pts) - 2), 6), np.float32)
    k = 0
    for a in range(1, len(pts) - 1):
        for idx in (0, a, a + 1):
            out[k, :2] = pts[idx]
            out[k, 2:] = col
            k += 1
    return out


def _segments(pts, col, closed=True):
    m = len(pts)
    pairs = m if closed else m - 1
    out = np.empty((2 * pairs, 6), np.float32)
    k = 0
    for i in range(pairs):
        a, b = pts[i], pts[(i + 1) % m]
        out[k, :2] = a; out[k, 2:] = col; k += 1
        out[k, :2] = b; out[k, 2:] = col; k += 1
    return out


class GLTrackWidget(CameraMixin, QOpenGLWidget):
    def __init__(self, world, parent=None):
        super().__init__(parent)
        self._init_camera(world)
        self.show_rays = True
        self.show_racing = True
        self.hud_font = QFont()
        self.hud_font.setPointSize(11)
        self.hud_font.setBold(True)
        self.setMinimumSize(500, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._prog = QOpenGLShaderProgram(self)
        self._vao = None
        self._tris, self._walls, self._lines = QOpenGLBuffer(), QOpenGLBuffer(), QOpenGLBuffer()
        self._dcars, self._drays, self._dlines = QOpenGLBuffer(), QOpenGLBuffer(), QOpenGLBuffer()
        self._n_tris = self._n_walls = self._n_lines = 0
        self._built_tv, self._built_race, self._built_ref = -1, None, None

    def initializeGL(self):
        try:
            from PySide6.QtOpenGL import QOpenGLVertexArrayObject
            self._vao = QOpenGLVertexArrayObject(self)
            self._vao.create()
            self._vao.bind()
        except Exception:
            self._vao = None
        st = getattr(QOpenGLShader, "ShaderTypeBit", QOpenGLShader)
        self._prog.addShaderFromSourceCode(st.Vertex, _VS)
        self._prog.addShaderFromSourceCode(st.Fragment, _FS)
        self._prog.bindAttributeLocation("pos", 0)
        self._prog.bindAttributeLocation("col", 1)
        self._prog.link()
        f = self.context().functions()
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _upload(self, buf, arr, usage):
        if arr.shape[0] == 0:
            return 0
        if not buf.isCreated():
            buf.create()
        buf.bind()
        buf.setUsagePattern(usage)
        buf.allocate(arr.tobytes(), arr.nbytes)
        buf.release()
        return arr.shape[0]

    def _build_static(self, t):
        o = np.array([(p.x(), p.y()) for p in t.poly_outer], np.float32)
        i = np.array([(p.x(), p.y()) for p in t.poly_inner], np.float32)
        n = len(o)
        road = np.empty((6 * n, 6), np.float32)
        k = 0
        for a in range(n):
            b = (a + 1) % n
            for tri in ((o[a], i[a], i[b]), (o[a], i[b], o[b])):
                for pt in tri:
                    road[k, :2] = pt
                    road[k, 2:] = ASPHALT
                    k += 1
        sx, sy, sang = t.start
        ca, sa = math.cos(sang), math.sin(sang)
        fx, fy, nx, ny = ca, sa, -sa, ca
        hw = t.half_width
        sq = (2.0 * hw) / 8.0
        checkers = np.empty((48, 6), np.float32)
        k = 0
        for q in range(8):
            d0, d1 = -hw + q * sq, -hw + (q + 1) * sq
            quad = [(sx - 3 * fx + nx * d0, sy - 3 * fy + ny * d0),
                    (sx - 3 * fx + nx * d1, sy - 3 * fy + ny * d1),
                    (sx + 3 * fx + nx * d1, sy + 3 * fy + ny * d1),
                    (sx + 3 * fx + nx * d0, sy + 3 * fy + ny * d0)]
            col = (0.93, 0.93, 0.94, 1.0) if q % 2 == 0 else (0.09, 0.09, 0.10, 1.0)
            for pt in (quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]):
                checkers[k, :2] = pt
                checkers[k, 2:] = col
                k += 1
        center = [(p.x(), p.y()) for p in t.center_poly]
        parts = [_segments(center, CENTER)]
        if self.show_racing and getattr(t, "racing", None):
            parts.append(_segments(t.racing, RACING))
        self._n_tris = self._upload(self._tris, np.concatenate([road, checkers]),
                                    QOpenGLBuffer.UsagePattern.StaticDraw)
        self._n_walls = self._upload(self._walls,
                                     np.concatenate([_segments(o, WALL), _segments(i, WALL)]),
                                     QOpenGLBuffer.UsagePattern.StaticDraw)
        self._n_lines = self._upload(self._lines, np.concatenate(parts),
                                     QOpenGLBuffer.UsagePattern.StaticDraw)

    def _dynamic(self, w):
        cars, rays, lines = [], [], []
        lead = w.leader()
        for car in (w.player, lead):
            if car is not None and car.alive and len(car.trail) > 1:
                col = ((1.0, 0.59, 0.24, 0.55) if car is w.player
                       else _hue_rgb(car.hue, 0.5, 1.0, 0.42))
                lines.append(_segments(car.trail, col, closed=False))
        for c in w.cars:
            if not c.alive and c.death_x is not None:
                ch, sh = math.cos(c.head), math.sin(c.head)
                cars.append(_fan([(c.death_x + ch * px - sh * py,
                                   c.death_y + sh * px + ch * py) for px, py in CAR_PTS], DEAD))
        if w.ghost is not None:
            g = w.ghost
            ch, sh = math.cos(g.head), math.sin(g.head)
            cars.append(_fan([(g.x + ch * px - sh * py,
                               g.y + sh * px + ch * py) for px, py in CAR_PTS], GHOST_C))
        for car in (lead, w.player):
            if car is not None and car.alive and self.show_rays:
                for off, r in zip(nr.RAY_ANGLES, car.rays):
                    ang = car.head + math.radians(off)
                    d = r * nr.RAY_RANGE
                    col = ((1 - 0.62 * r, 0.27 + 0.69 * r, 0.27, 0.30) if car is lead
                           else (0.55, 0.78, 1.0, 0.24))
                    rays.append(np.array([[car.x, car.y, *col],
                                          [car.x + math.cos(ang) * d,
                                           car.y + math.sin(ang) * d, *col]], np.float32))
        for c in w.cars:
            if not c.alive:
                continue
            ch, sh = math.cos(c.head), math.sin(c.head)
            cars.append(_fan([(c.x + ch * px - sh * py, c.y + sh * px + ch * py)
                              for px, py in CAR_PTS],
                             _hue_rgb(c.hue, 0.6, 1.0 if c is lead else 0.85)))
        pl = w.player
        if pl is not None and pl.alive:
            ch, sh = math.cos(pl.head), math.sin(pl.head)
            cars.append(_fan([(pl.x + ch * px - sh * py, pl.y + sh * px + ch * py)
                              for px, py in CAR_PTS], PLAYER))
        return cars, rays, lines

    def _draw(self, buf, mode, count):
        if count <= 0 or not buf.isCreated():
            return
        buf.bind()
        self._prog.enableAttributeArray(0)
        self._prog.setAttributeBuffer(0, GL_FLOAT, 0, 2, 24)
        self._prog.enableAttributeArray(1)
        self._prog.setAttributeBuffer(1, GL_FLOAT, 8, 4, 24)
        self.context().functions().glDrawArrays(mode, 0, count)
        self._prog.disableAttributeArray(0)
        self._prog.disableAttributeArray(1)
        buf.release()

    def paintGL(self):
        w = self.replay if self.replay is not None else self.world
        t = w.track
        f = self.context().functions()
        f.glClearColor(0.058, 0.066, 0.082, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT)
        pl0, lead0 = self.update_camera()
        if (self._built_tv != w.track_version or self._built_race != self.show_racing
                or self._built_ref is not t):
            self._build_static(t)
            self._built_tv, self._built_race, self._built_ref = w.track_version, self.show_racing, t
        if not self._prog.bind():
            return
        self._prog.setUniformValue("res", QVector2D(self.width(), self.height()))
        self._prog.setUniformValue("cam", QVector2D(float(self.camx), float(self.camy)))
        self._prog.setUniformValue("scale", float(self._scale()))
        self._draw(self._tris, GL_TRIANGLES, self._n_tris)
        if self._n_lines:
            f.glLineWidth(1.5)
            self._draw(self._lines, GL_LINES, self._n_lines)
        if self._n_walls:
            f.glLineWidth(2.4)
            self._draw(self._walls, GL_LINES, self._n_walls)
        cars, rays, lines = self._dynamic(w)
        if cars:
            n = self._upload(self._dcars, np.concatenate(cars), QOpenGLBuffer.UsagePattern.StreamDraw)
            self._draw(self._dcars, GL_TRIANGLES, n)
        if rays:
            n = self._upload(self._drays, np.concatenate(rays), QOpenGLBuffer.UsagePattern.StreamDraw)
            f.glLineWidth(1.2)
            self._draw(self._drays, GL_LINES, n)
        if lines:
            n = self._upload(self._dlines, np.concatenate(lines), QOpenGLBuffer.UsagePattern.StreamDraw)
            f.glLineWidth(2.0)
            self._draw(self._dlines, GL_LINES, n)
        self._prog.release()

        p = QPainter(self)
        p.setFont(self.hud_font)
        txt, extra = self._hud_texts(w, lead0, pl0)
        fm = p.fontMetrics()
        p.fillRect(QRectF(10, 10, fm.horizontalAdvance(txt) + 20, 28), QColor(0, 0, 0, 130))
        p.fillRect(QRectF(10, 42, max(10.0, fm.horizontalAdvance(extra) + 20.0), 22),
                   QColor(0, 0, 0, 130))
        p.setPen(QColor(232, 235, 242))
        p.drawText(QPointF(20, 29), txt)
        p.setPen(QColor(255, 190, 120) if pl0 is not None else QColor(170, 176, 190))
        p.drawText(QPointF(20, 58), extra)
        p.end()

        if self.editor_mode:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            s = self._scale()
            p.translate(self.width() / 2, self.height() / 2)
            p.scale(s, s)
            p.translate(-self.camx, -self.camy)
            self._paint_editor_world(p, s)
            p.resetTransform()
            self._paint_editor_hud(p)
            p.end()
