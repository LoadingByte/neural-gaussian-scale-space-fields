import math
import time
from pathlib import Path

import cv2
import glfw
import numpy as np
import trimesh
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

res = (1800, 1000)
max_fps = 60
near = 0.1
far = 100.0
fov = 0.8
mov_speed = 0.5
rot_speed = 0.005
gizmo_and_grid = False

shader_dir = Path(__file__).parent / "shaders"


class Window:

    def __init__(self):
        glfw.init()
        self.window_handle = glfw.create_window(*res, "Smoothable Neural Texture", None, None)
        if not self.window_handle:
            raise ValueError("Unable to create GLFW window.")
        glfw.make_context_current(self.window_handle)
        glfw.set_key_callback(self.window_handle, self.key_callback)
        glfw.set_input_mode(self.window_handle, glfw.CURSOR, glfw.CURSOR_DISABLED)

        self.display_shader = compileProgram(
            compileShader((shader_dir / "display.vert").read_text(), GL_VERTEX_SHADER),
            compileShader((shader_dir / "display.frag").read_text(), GL_FRAGMENT_SHADER)
        )
        self.mesh_shader = compileProgram(
            compileShader((shader_dir / "mesh.vert").read_text(), GL_VERTEX_SHADER),
            compileShader((shader_dir / "mesh.frag").read_text(), GL_FRAGMENT_SHADER)
        )

        self.color, self.depth = glGenTextures(2)
        glBindTexture(GL_TEXTURE_2D, self.color)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, *res, 0, GL_RGBA, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, self.depth)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, *res, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

        self.fbo = glGenFramebuffers(1)

        self.setup_mesh()

        self.is_recording = False
        self.recorded_frames = []
        self.frames = 0
        self.camX = 0.0
        self.camY = 0.0
        self.camZ = 2.0
        self.camPitch = 0.0
        self.camYaw = 0.0
        self.prevCursorX = 0.0
        self.prevCursorY = 0.0

    def setup_mesh(self):
        mesh = trimesh.load_mesh(Path(__file__).parent.parent / "data" / "textured" / "fish" / "model.obj")
        vertices = (mesh.vertices - mesh.bounding_box.centroid) * (2 / np.max(mesh.bounding_box.extents))
        self.num_faces = mesh.faces.shape[0]

        self.vao = glGenVertexArrays(1)
        vertex_pos_buf, vertex_uv_buf, index_buf = glGenBuffers(3)

        glBindVertexArray(self.vao)

        position_attr = glGetAttribLocation(self.mesh_shader, "position")
        glBindBuffer(GL_ARRAY_BUFFER, vertex_pos_buf)
        glEnableVertexAttribArray(position_attr)
        glVertexAttribPointer(position_attr, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, vertices.astype(np.float32), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        uv_attr = glGetAttribLocation(self.mesh_shader, "uv")
        glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buf)
        glEnableVertexAttribArray(uv_attr)
        glVertexAttribPointer(uv_attr, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glBufferData(GL_ARRAY_BUFFER, mesh.visual.uv.astype(np.float32), GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buf)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.faces.astype(np.uint32), GL_STATIC_DRAW)

        glBindVertexArray(0)

    def run(self):
        while not glfw.window_should_close(self.window_handle):
            start_time = time.perf_counter()

            self.move_camera()

            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            self.render()
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            if self.is_recording:
                self.recorded_frames.append(self.download_image())

            # Display self.color on screen.
            glViewport(0, 0, *res)
            glClear(GL_COLOR_BUFFER_BIT)
            glDisable(GL_CULL_FACE)
            glDisable(GL_DEPTH_TEST)
            glUseProgram(self.display_shader)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.color)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0)
            glVertex2f(-1, -1)
            glTexCoord2f(0, 1)
            glVertex2f(1, -1)
            glTexCoord2f(1, 1)
            glVertex2f(1, 1)
            glTexCoord2f(1, 0)
            glVertex2f(-1, 1)
            glEnd()
            glBindTexture(GL_TEXTURE_2D, 0)
            glUseProgram(0)

            glfw.swap_buffers(self.window_handle)
            glfw.poll_events()

            render_time = time.perf_counter() - start_time
            if render_time < (1 / max_fps):
                time.sleep((1 / max_fps) - render_time)

        glfw.terminate()

    def move_camera(self):
        delta = 1 / max_fps
        if glfw.get_key(self.window_handle, glfw.KEY_W) == glfw.PRESS:
            self.camX += math.sin(self.camYaw) * delta * mov_speed
            self.camZ -= math.cos(self.camYaw) * delta * mov_speed
        if glfw.get_key(self.window_handle, glfw.KEY_S) == glfw.PRESS:
            self.camX -= math.sin(self.camYaw) * delta * mov_speed
            self.camZ += math.cos(self.camYaw) * delta * mov_speed
        if glfw.get_key(self.window_handle, glfw.KEY_A) == glfw.PRESS:
            self.camX -= math.cos(self.camYaw) * delta * mov_speed
            self.camZ -= math.sin(self.camYaw) * delta * mov_speed
        if glfw.get_key(self.window_handle, glfw.KEY_D) == glfw.PRESS:
            self.camX += math.cos(self.camYaw) * delta * mov_speed
            self.camZ += math.sin(self.camYaw) * delta * mov_speed
        if glfw.get_key(self.window_handle, glfw.KEY_SPACE) == glfw.PRESS:
            self.camY += delta * mov_speed
        if glfw.get_key(self.window_handle, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.camY -= delta * mov_speed

        cursorX, cursorY = glfw.get_cursor_pos(self.window_handle)
        # All kind of jitter happens in the first few frames.
        self.frames += 1
        if self.frames > 10:
            self.camPitch = min(
                max(self.camPitch - (cursorY - self.prevCursorY) * rot_speed, -math.pi / 2),
                math.pi / 2
            )
            self.camYaw += (cursorX - self.prevCursorX) * rot_speed
        self.prevCursorX = cursorX
        self.prevCursorY = cursorY

    def render(self):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth, 0)

        glViewport(0, 0, *res)
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspectRatio = res[0] / res[1]
        glFrustum(-near * fov, near * fov, -near * fov / aspectRatio, near * fov / aspectRatio, near, far)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotated(math.degrees(self.camPitch), -1, 0, 0)
        glRotated(math.degrees(self.camYaw), 0, 1, 0)
        glTranslated(-self.camX, -self.camY, -self.camZ)

        if gizmo_and_grid:
            self.render_gizmo()
            self.render_grid()
        self.render_mesh()

    @staticmethod
    def render_gizmo():
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3d(1, 0, 0)
        glVertex3d(0, 0, 0)
        glVertex3d(1, 0, 0)
        glColor3d(0, 1, 0)
        glVertex3d(0, 0, 0)
        glVertex3d(0, 1, 0)
        glColor3d(0, 0, 1)
        glVertex3d(0, 0, 0)
        glVertex3d(0, 0, 1)
        glEnd()

    @staticmethod
    def render_grid():
        n_lines = 5
        line_spacing = 1
        glColor3d(0.5, 0.5, 0.5)
        glLineWidth(1)
        glBegin(GL_LINES)
        for i in range(-n_lines, n_lines + 1):
            if i == 0:
                # Leave gaps for the gizmo.
                # X
                glVertex3d(-n_lines * line_spacing, 0, i * line_spacing)
                glVertex3d(0, 0, i * line_spacing)
                glVertex3d(1, 0, i * line_spacing)
                glVertex3d(n_lines * line_spacing, 0, i * line_spacing)
                # Z
                glVertex3d(i * line_spacing, 0, -n_lines * line_spacing)
                glVertex3d(i * line_spacing, 0, 0)
                glVertex3d(i * line_spacing, 0, 1)
                glVertex3d(i * line_spacing, 0, n_lines * line_spacing)
            else:
                # X
                glVertex3d(-n_lines * line_spacing, 0, i * line_spacing)
                glVertex3d(n_lines * line_spacing, 0, i * line_spacing)
                # Z
                glVertex3d(i * line_spacing, 0, -n_lines * line_spacing)
                glVertex3d(i * line_spacing, 0, n_lines * line_spacing)
        glEnd()

    def render_mesh(self):
        glPushMatrix()
        glRotated(-90, 1, 0, 0)
        glUseProgram(self.mesh_shader)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.num_faces * 3, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)
        glUseProgram(0)
        glPopMatrix()

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.window_handle, True)
            elif key == glfw.KEY_C:
                d = Path(__file__).parent.parent / "results" / "textured" / "fish" / "screenshots" / "uv"
                i = (max((int(f.stem) for f in d.iterdir()), default=-1) + 1) if d.exists() else 0
                self.save_capture(d / f"{i:05}.png", self.download_image())
            elif key == glfw.KEY_R:
                if self.is_recording:
                    d = Path(__file__).parent.parent / "results" / "textured" / "fish" / "moving" / "uv"
                    for i, frame in enumerate(self.recorded_frames):
                        self.save_capture(d / f"{i:05}.png", frame)
                    self.recorded_frames.clear()
                self.is_recording = not self.is_recording

    @staticmethod
    def save_capture(file, cap):
        file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(file), (cv2.cvtColor(cap, cv2.COLOR_RGBA2BGRA).clip(0, 1) * 65535).astype(np.uint16))

    def download_image(self):
        glBindTexture(GL_TEXTURE_2D, self.color)
        image = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)
        glBindTexture(GL_TEXTURE_2D, 0)
        return np.flip(np.reshape(image, (res[1], res[0], 4)), 0)


if __name__ == "__main__":
    Window().run()
