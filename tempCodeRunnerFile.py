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

FRAG_SHADER = """
#version 330
in vec2 uv;
out vec4 fragColor;

uniform sampler2D image;
uniform float progress;
uniform float time;
uniform float isPaused;

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
    
    // Si está pausado, solo mostrar la imagen base
    if (isPaused > 0.5) {
        fragColor = vec4(color, 1.0);
        return;
    }
 
    // 1. Horizontal Line Displacement (estilo datamosh)
    float lineHeight = 2.0 + progress * 8.0;
    float lineIndex = floor(uv.y * height / lineHeight);
    float displaceAmount = (rand(vec2(lineIndex, floor(t * 2.0))) - 0.5) * 0.3 * pow(progress, 0.6);
    float shouldDisplace = step(0.7, rand(vec2(lineIndex, floor(t * 3.0))));
    vec2 displacedUV = vec2(uv.x + displaceAmount * shouldDisplace, uv.y);
    color = texture(image, displacedUV).rgb;

    // 2. Pixel Sorting / Stretching
    float sortLine = floor(uv.y * (50.0 + progress * 200.0));
    float sortTrigger = step(0.75, rand(vec2(sortLine, floor(t * 2.5))));
    if (sortTrigger > 0.5) {
        float stretchX = uv.x + (rand(vec2(sortLine, t)) - 0.5) * 0.1;
        vec2 stretchUV = vec2(stretchX, uv.y);
        vec3 stretchedColor = texture(image, stretchUV).rgb;
        color = mix(color, stretchedColor, pow(progress, 0.7));
    }

    // 3. RGB Channel Shift (separación extrema de canales)
    float shiftAmount = 0.02 + 0.08 * pow(progress, 0.8);
    float shiftPattern = floor(uv.y * (30.0 + progress * 100.0));
    float shiftR = (rand(vec2(shiftPattern, floor(t * 2.0))) - 0.5) * shiftAmount;
    float shiftG = (rand(vec2(shiftPattern + 1.0, floor(t * 2.0))) - 0.5) * shiftAmount;
    float shiftB = (rand(vec2(shiftPattern + 2.0, floor(t * 2.0))) - 0.5) * shiftAmount;
    
    float r = texture(image, vec2(uv.x + shiftR, uv.y)).r;
    float g = texture(image, vec2(uv.x + shiftG, uv.y)).g;
    float b = texture(image, vec2(uv.x + shiftB, uv.y)).b;
    
    color = vec3(r, g, b);

    // Clamp + output
    color = clamp(color, 0.0, 1.0);
    fragColor = vec4(color, 1.0);
}
"""

UI_VERT_SHADER = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main() {
    uv = (in_vert + 1.0) * 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

UI_FRAG_SHADER = """
#version 330
in vec2 uv;
out vec4 fragColor;

uniform vec2 resolution;
uniform vec2 buttonPos;
uniform vec2 buttonSize;
uniform float isPaused;

float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    vec2 pixelCoord = uv * resolution;
    vec2 center = buttonPos + buttonSize * 0.5;
    vec2 p = pixelCoord - center;
    
    float cornerRadius = 12.0;
    float d = sdRoundBox(p, buttonSize * 0.5, cornerRadius);
    
    // Botón base
    float button = smoothstep(1.0, -1.0, d);
    
    // Color de fondo simple
    vec3 bgColor = vec3(0.18, 0.18, 0.20);
    
    // Borde sutil
    float border = smoothstep(2.0, 0.0, abs(d)) - smoothstep(1.0, 0.0, abs(d));
    vec3 borderColor = vec3(0.35, 0.35, 0.40);
    
    // Símbolo
    vec3 symbolColor = vec3(0.95);
    float symbol = 0.0;
    
    if (isPaused > 0.5) {
        // Triángulo de play (apuntando a la derecha)
        vec2 tp = p / buttonSize;
        // Crear un triángulo simple: punto en X>0, se estrecha en Y
        float slope = 0.12 / 0.12; // altura / base
        float leftEdge = -0.06; // inicio del triángulo
        float rightEdge = 0.06; // punta del triángulo
        
        // Dentro del triángulo si:
        // 1. X está entre leftEdge y rightEdge
        // 2. Y está dentro de las líneas diagonales
        float inX = step(leftEdge, tp.x) * step(tp.x, rightEdge);
        float maxY = slope * (rightEdge - tp.x);
        float inY = step(-maxY, tp.y) * step(tp.y, maxY);
        
        symbol = inX * inY;
    } else {
        // Barras de pausa
        vec2 bp = p / buttonSize;
        float bar1 = step(abs(bp.x + 0.06), 0.03) * step(abs(bp.y), 0.12);
        float bar2 = step(abs(bp.x - 0.06), 0.03) * step(abs(bp.y), 0.12);
        symbol = max(bar1, bar2);
    }
    
    // Combinar
    vec3 color = bgColor;
    color = mix(color, symbolColor, symbol);
    color = mix(color, borderColor, border);
    
    float alpha = max(button, border) * 0.9;
    fragColor = vec4(color, alpha);
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
    width, height = 800, 600
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Internal Noise - Arte y Tecnología")

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    
    program = ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)
    ui_program = ctx.program(vertex_shader=UI_VERT_SHADER, fragment_shader=UI_FRAG_SHADER)

    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(program, vbo, 'in_vert')
    ui_vao = ctx.simple_vertex_array(ui_program, vbo, 'in_vert')

    texture = load_texture(ctx, "demo.jpg")
    texture.use(location=0)
    program['image'] = 0
    program['height'] = float(height)

    start_time = time.time()
    paused_time = 0.0
    duration = 60.0
    clock = pygame.time.Clock()
    
    is_paused = False
    pause_start = 0.0
    
    button_pos = np.array([width - 80.0, height - 80.0], dtype='f4')
    button_size = np.array([60.0, 60.0], dtype='f4')

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_r:
                    start_time = time.time()
                    paused_time = 0.0
                    is_paused = False
                elif event.key == K_SPACE:
                    is_paused = not is_paused
                    if is_paused:
                        pause_start = time.time()
                    else:
                        start_time = time.time()
                        paused_time = 0.0
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  
                    mx, my = event.pos
                    my = height - my 
                    if (button_pos[0] <= mx <= button_pos[0] + button_size[0] and
                        button_pos[1] <= my <= button_pos[1] + button_size[1]):
                        is_paused = not is_paused
                        if is_paused:
                            pause_start = time.time()
                        else:
                            start_time = time.time()
                            paused_time = 0.0

        if is_paused:
            elapsed = pause_start - start_time - paused_time
        else:
            elapsed = time.time() - start_time - paused_time
            
        progress = min(1.0, elapsed / duration)

        ctx.clear(0.0, 0.0, 0.0)
        
        program['progress'].value = progress
        program['time'].value = elapsed
        program['isPaused'].value = 1.0 if is_paused else 0.0
        vao.render(moderngl.TRIANGLE_STRIP)

        ui_program['resolution'].value = (width, height)
        ui_program['buttonPos'].value = tuple(button_pos)
        ui_program['buttonSize'].value = tuple(button_size)
        ui_program['isPaused'].value = 1.0 if is_paused else 0.0
        ui_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()