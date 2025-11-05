import pygame
from pygame.locals import *
import moderngl
import numpy as np
from PIL import Image
import time

VERT_SHADER = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main() {
    uv = (in_vert + 1.0) * 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

# ToDo: HACER NUESTROS PROPIOS FRAGMENT SHADERS QUE VAYAN AUMENTANDO CON EL PASO DEL TIEMPO
FRAG_SHADER = """
#version 330
in vec2 uv;
out vec4 fragColor;

uniform sampler2D image;
uniform float progress;
uniform float time;

// Random + noise functions
float rand(vec2 co) {
    return fract(sin(dot(co.xy , vec2(12.9898,78.233))) * 43758.5453);
}

float smoothNoise(vec2 uv) {
    vec2 i = floor(uv);
    vec2 f = fract(uv);
    float a = rand(i);
    float b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));
    vec2 u = f*f*(3.0 - 2.0*f);
    return mix(a, b, u.x) + (c - a)*u.y*(1.0 - u.x) + (d - b)*u.x*u.y;
}

void main() {
    vec2 uvn = uv;
    float t = time;

    // Base calm image
    vec3 base = texture(image, uv).rgb;
    vec3 color = base;
 
    // 1. Grain + flicker (always increasing)
    float grain = rand(uv * (300.0 + progress * 1200.0) + t * 10.0);
    float flicker = 1.0 + 0.3 * sin(t * (2.0 + progress * 8.0));
    color *= flicker;
    color += (grain - 0.5) * 0.2 * progress;

    // 2. Infection spread (expanding irregular mask)
    float spread = pow(progress, 0.6);
    float n = smoothNoise(uv * (6.0 + progress * 40.0) + t * 0.5);
    float mask = smoothstep(spread - 0.1, spread + 0.1, 1.0 - min(min(uv.x, 1.0-uv.x), min(uv.y, 1.0-uv.y)) + (n-0.5)*0.5);
    vec3 infection = mix(color, vec3(n), mask * 0.8 * progress);
    color = mix(color, infection, progress);

    // 3. Chromatic aberration (amplitude grows)
    float chroma = 0.02 * progress * (0.5 + 0.5 * sin(t * 0.7));
    vec2 offset = vec2(chroma * (rand(vec2(t, uv.y)) - 0.5), 0.0);
    float r = texture(image, uv + offset).r;
    float g = texture(image, uv).g;
    float b = texture(image, uv - offset).b;
    vec3 abColor = vec3(r, g, b);
    color = mix(color, abColor, progress * 1.2);

    // 4. Melting distortion (waves + time)
    float meltStrength = 0.005 + 0.02 * progress;
    vec2 meltUV = uv + vec2(
        sin(uv.y * (40.0 + 100.0 * progress) + t * (1.0 + progress * 5.0)) * meltStrength,
        cos(uv.x * (30.0 + 80.0 * progress) + t * (0.5 + progress * 3.0)) * meltStrength
    );
    color = mix(color, texture(image, meltUV).rgb, progress * 0.8);

    // 5. Glitch slicing (horizontal/vertical shifts)
    float glitchFreq = 0.2 + progress * 5.0;
    float glitchLine = step(0.9, rand(vec2(floor(uv.y * (10.0 + progress * 100.0)), t * glitchFreq)));
    vec2 gShift = vec2(glitchLine * (0.05 + 0.15 * progress) * (rand(vec2(uv.y, t)) - 0.5), 0.0);
    vec3 glitchColor = texture(image, uv + gShift).rgb;
    color = mix(color, glitchColor, glitchLine * progress);

    // 6. Dissolution into full noise
    float dissolve = pow(progress, 2.0);
    float fullNoise = rand(uv * (500.0 + progress * 5000.0) + t * 20.0);
    vec3 dissolved = mix(color, vec3(fullNoise), dissolve);
    color = mix(color, dissolved, dissolve);

    // Clamp + output
    color = clamp(color, 0.0, 1.0);
    fragColor = vec4(color, 1.0);
}
"""

def load_texture(ctx, path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
    tex = ctx.texture(img.size, 3, img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex

def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Internal Noise - Arte y Tecnología")

    ctx = moderngl.create_context()
    program = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)

    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(program, vbo, 'in_vert')

    texture = load_texture(ctx, "demo.jpg")
    texture.use(location=0)
    program['image'] = 0

    start_time = time.time()
    duration = 60.0
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / duration)

        ctx.clear(0.0, 0.0, 0.0)
        program['progress'].value = progress
        program['time'].value = elapsed
        vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
