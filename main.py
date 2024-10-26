"""
    Toy Path Tracer Project
    https://github.com/kadir014/toy-pathtracer
"""

from array import array

import pygame
import moderngl


WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
MAX_FPS = 240

CAM_MOV_SPEED = 20             # Camera linear speed
CAM_ROT_SPEED = 100            # Camera rotation speed
CAM_FWD =       pygame.K_w     # Move camera forward
CAM_BACK =      pygame.K_s     # Move camera backwards
CAM_LEFT =      pygame.K_a     # Strafe camera to left
CAM_RIGHT =     pygame.K_d     # Strafe camera to right
CAM_UP =        pygame.K_q     # Move camera upward
CAM_DOWN =      pygame.K_e     # Move camera downward
CAM_ROT_LEFT =  pygame.K_LEFT  # Rotate camera to left
CAM_ROT_RIGHT = pygame.K_RIGHT # Rotate camera to right
CAM_ROT_UP =    pygame.K_UP    # Rotate camera to up
CAM_ROT_DOWN =  pygame.K_DOWN  # Rotate camera to down
CAM_FL_INC =    pygame.K_j     # Camera focal length increase
CAM_FL_DEC =    pygame.K_k     # Camera focal length decrease
EXPOSURE_INC =  pygame.K_z     # Increase exposure
EXPOSURE_DEC =  pygame.K_x     # Decrease exposure

QUIT            = pygame.K_ESCAPE # Quit the app
TOGGLE_ROULETTE = pygame.K_F2     # Enable/disable russian roulette


class Camera:
    """
    A simple camera model using a physical camera setup defined 
    by lens focal length and image horizontal size.

    u and v attributes are the directions on the image plane for shooting rays.
    """

    def __init__(self,
            viewport: tuple[int, int],
            position: tuple[float, float, float] | pygame.Vector3 = (0.0, 0.0, 0.0)
            ) -> None:
        self.aspect_ratio = viewport[0] / viewport[1]
        self.position = pygame.Vector3(position)
        self.look_at = pygame.Vector3(0.0, 0.0, 1.0)
        self.up = pygame.Vector3(0.0, 1.0, 0.0)

        self.focal_length = 0.15
        self.horizontal_size = 0.16

        self.update()

    def update(self) -> None:
        """ Calculate and update camera vectors. """

        alignment = (self.look_at - self.position).normalize()

        self.u = alignment.cross(self.up).normalize()
        self.v = self.u.cross(alignment).normalize()

        self.center = self.position + self.focal_length * alignment

        self.u *= self.horizontal_size
        self.v *= self.horizontal_size / self.aspect_ratio

    def move(self, amount: float) -> None:
        """ Move in camera's current direction. """

        delta = (self.look_at - self.position).normalize() * amount
        self.position += delta
        self.look_at += delta

    def strafe(self, amount: float) -> None:
        """ Strafe horizontally. """

        delta = (self.look_at - self.position).normalize()
        delta = delta.cross(pygame.Vector3(0.0, 1.0, 0.0)) * amount
        self.position += delta
        self.look_at += delta

    def rotate_yaw(self, amount: float) -> None:
        """ Rotate camera horizontally. """

        delta = self.look_at - self.position
        delta = delta.rotate(amount, self.up)
        self.look_at = self.position + delta

    def rotate_pitch(self, amount: float) -> None:
        """ Rotate camera vertically. """

        delta = self.look_at - self.position
        delta = delta.rotate(amount, self.up.cross(delta.normalize()))
        self.look_at = self.position + delta


class PathTracerEngine:
    """
    Path tracer rendering engine.
    """

    def __init__(self) -> None:
        self._context = moderngl.create_context()

        self.frame = 0
        self.true_frame = 0

        self.camera = Camera((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.camera.position = pygame.Vector3(-11.7752, -8.55655, 19.0065)
        self.camera.look_at = pygame.Vector3(-11.3298, -8.10723, 19.78095)
        self.camera.update()

        base_vertex_shader = """
        #version 330

        in vec2 in_position;
        in vec2 in_uv;

        out vec2 v_uv;

        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);

            v_uv = in_uv;
        }
        """

        self._display_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=open("display.fsh").read()
        )

        self.exposure = 0.5

        self._pathtracer_program = self._context.program(
            vertex_shader=base_vertex_shader,
            fragment_shader=open("pathtracer.fsh").read()
        )

        self._update_camera_uniform()

        self._pathtracer_program["s_sky"] = 0
        self._pathtracer_program["s_prev"] = 1
        self._pathtracer_program["u_viewport"] = (WINDOW_WIDTH, WINDOW_HEIGHT)

        self.russian_roulette = True

        vbo = self.create_buffer_object([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        uvbo = self.create_buffer_object([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        ibo = self.create_buffer_object( [0, 1, 2, 1, 2, 3])

        # Using the same buffers for VAOs since they won't change

        self._pathtracer_vao = self._context.vertex_array(
            self._pathtracer_program,
            (
                (vbo, "2f", "in_position"),
                (uvbo, "2f", "in_uv")
            ),
            ibo
        )

        self._display_vao = self._context.vertex_array(
            self._display_program,
            (
                (vbo, "2f", "in_position"),
                (uvbo, "2f", "in_uv")
            ),
            ibo
        )

        # Note: data type being f4 lets us store colors in HDR
        self._pathtracer_target_texture = self._context.texture((WINDOW_WIDTH, WINDOW_HEIGHT), 4, dtype="f4")
        self._pathtracer_fbo = self._context.framebuffer(color_attachments=(self._pathtracer_target_texture,))

        self.load_sky(pygame.Surface((1, 1)))

    def __del__(self) -> None:
        self._context.release()

    def create_buffer_object(self, data: list) -> moderngl.Buffer:
        """ Create buffer object from array. """

        dtype = "f" if isinstance(data[0], float) else "I"
        return self._context.buffer(array(dtype, data))
    
    def load_sky(self, surface: pygame.Surface) -> None:
        """ Load sky texture. """

        self._sky_texture = self._context.texture(
            surface.get_size(),
            3,
            pygame.image.tobytes(surface, "RGB")
        )

    def reset_accumulation(self) -> None:
        """ Reset frame accumulation used for progressive rendering. """

        self._pathtracer_fbo.clear()
        self._update_camera_uniform()
        self.frame = 0

    def render(self) -> None:
        """ Render one frame. """

        self._pathtracer_fbo.use()
        self._pathtracer_program["u_frame"] = self.frame
        self._pathtracer_program["u_true_frame"] = self.true_frame
        self._sky_texture.use(0)
        self._pathtracer_target_texture.use(1)
        self._pathtracer_vao.render()

        self._context.screen.use()
        self._context.clear()
        self._pathtracer_target_texture.use(0)
        self._display_vao.render()

        self.frame += 1
        self.true_frame += 1

    def _update_camera_uniform(self) -> None:
        self._pathtracer_program["u_camera.position"] = (
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        )
        self._pathtracer_program["u_camera.center"] = (
            self.camera.center.x,
            self.camera.center.y,
            self.camera.center.z
        )
        self._pathtracer_program["u_camera.u"] = (
            self.camera.u.x,
            self.camera.u.y,
            self.camera.u.z
        )
        self._pathtracer_program["u_camera.v"] = (
            self.camera.v.x,
            self.camera.v.y,
            self.camera.v.z
        )

    @property
    def russian_roulette(self) -> bool:
        return self.__roulette
    
    @russian_roulette.setter
    def russian_roulette(self, val: bool) -> None:
        self.__roulette = val
        self._pathtracer_program["u_roulette"] = self.__roulette

    @property
    def exposure(self) -> float:
        return self.__exposure
    
    @exposure.setter
    def exposure(self, val: float) -> None:
        self.__exposure = val
        self._display_program["u_exposure"] = self.__exposure


if __name__ == "__main__":
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    is_running = True

    engine = PathTracerEngine()

    while is_running:
        dt = clock.tick(MAX_FPS) * 0.001
        fps = clock.get_fps()
        pygame.display.set_caption(
            f"Pygame & ModernGL Progressive Path Tracer @{round(fps)} FPS - "
            f"Frame {engine.true_frame} - "
            f"Exposure {round(engine.exposure, 3)}"
        )

        reset_accumulation = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.KEYDOWN:
                if event.key == QUIT:
                    is_running = False

                elif event.key == pygame.K_F1:
                    print(
                        "Camera\n"
                        f"  Position:     {engine.camera.position}\n"
                        f"  Look At:      {engine.camera.look_at}\n"
                        f"  Up:           {engine.camera.up}\n"
                        f"  Focal Length: {engine.camera.focal_length}\n"
                        f"  Hor. Size:    {engine.camera.horizontal_size}\n"
                    )

                elif event.key == TOGGLE_ROULETTE:
                    engine.russian_roulette = not engine.russian_roulette
                    reset_accumulation = True

        keys = pygame.key.get_pressed()

        mov = CAM_MOV_SPEED * dt
        rot = CAM_ROT_SPEED * dt

        if keys[CAM_FWD]:
            engine.camera.move(mov)

        if keys[CAM_BACK]:
            engine.camera.move(-mov)

        if keys[CAM_LEFT]:
            engine.camera.strafe(-mov)

        if keys[CAM_RIGHT]:
            engine.camera.strafe(mov)

        if keys[CAM_UP]:
            engine.camera.position -= pygame.Vector3(0.0, mov, 0.0)
            engine.camera.look_at -= pygame.Vector3(0.0, mov, 0.0)

        if keys[CAM_DOWN]:
            engine.camera.position += pygame.Vector3(0.0, mov, 0.0)
            engine.camera.look_at += pygame.Vector3(0.0, mov, 0.0)

        if keys[CAM_ROT_LEFT]:
            engine.camera.rotate_yaw(rot)

        if keys[CAM_ROT_RIGHT]:
            engine.camera.rotate_yaw(-rot)

        if keys[CAM_ROT_UP]:
            engine.camera.rotate_pitch(rot)

        if keys[CAM_ROT_DOWN]:
            engine.camera.rotate_pitch(-rot)

        if keys[CAM_FL_INC]:
            engine.camera.focal_length += 0.003

        if keys[CAM_FL_DEC]:
            engine.camera.focal_length -= 0.003

        if keys[EXPOSURE_INC]:
            engine.exposure += 0.0075

        if keys[EXPOSURE_DEC]:
            engine.exposure -= 0.0075
            if engine.exposure <= 0.0:
                engine.exposure = 0.0

        # Reset progressive rendering accumulation if camera is altered
        if keys[CAM_FWD] or keys[CAM_BACK] or keys[CAM_LEFT] or keys[CAM_RIGHT] or \
           keys[CAM_UP] or keys[CAM_DOWN] or keys[CAM_ROT_LEFT] or keys[CAM_ROT_RIGHT] or \
           keys[CAM_ROT_UP] or keys[CAM_ROT_DOWN] or keys[CAM_FL_INC] or keys[CAM_FL_DEC]:
            reset_accumulation = True
            engine.camera.update()
        
        if reset_accumulation:
            engine.reset_accumulation()

        engine.render()

        pygame.display.flip()

    pygame.quit()