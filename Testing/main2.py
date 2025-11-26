import pygame
from pygame.locals import *
import moderngl
import numpy as np
from PIL import Image
import time
import os

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
uniform bool isPaused;

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
    if (isPaused) {
        fragColor = vec4(color, 1.0);
        return;
    }

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
    
    // 7. Horizontal scan lines (efecto de líneas de escaneo)
    float scanlineFreq = 300.0 + progress * 200.0;
    float scanline = sin(uv.y * scanlineFreq + t * 2.0) * 0.5 + 0.5;
    scanline = pow(scanline, 8.0) * 0.3 * progress;
    color += vec3(scanline);

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
uniform bool isPaused;

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
    
    if (!isPaused) {
        // Triángulo de play (apuntando a la derecha)
        vec2 tp = p / buttonSize;
        float slope = 0.12 / 0.12;
        float leftEdge = -0.06;
        float rightEdge = 0.06;
        
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

class AudioLayer:
    """Gestiona una capa de audio que se activa en un momento específico"""
    def __init__(self, path, activation_progress, volume=1.0):
        self.path = path
        self.activation_progress = activation_progress
        self.volume = volume
        self.sound = None
        self.channel = None
        self.is_active = False
        self.was_paused = False
        
    def load(self):
        """Carga el archivo de audio si existe"""
        if os.path.exists(self.path):
            try:
                self.sound = pygame.mixer.Sound(self.path)
                self.sound.set_volume(self.volume)
                return True
            except Exception as e:
                print(f"❌ Error cargando {self.path}: {e}")
                return False
        return False
            
    def update(self, current_progress, is_paused):
        """Actualiza el estado del audio según el progress y pausa"""
        if not self.sound:
            return
            
        # Activar el audio cuando se alcanza el punto de activación
        if current_progress >= self.activation_progress and not self.is_active:
            self.channel = self.sound.play(loops=-1)  # Loop infinito
            self.is_active = True
            print(f"🔊 Audio activado: {os.path.basename(self.path)} (progress: {current_progress:.2f})")
        
        # Gestionar pausa/reanudar
        if self.is_active and self.channel:
            if is_paused and not self.was_paused:
                self.channel.pause()
                self.was_paused = True
            elif not is_paused and self.was_paused:
                self.channel.unpause()
                self.was_paused = False
    
    def stop(self):
        """Detiene el audio y resetea el estado"""
        if self.channel:
            self.channel.stop()
        self.is_active = False
        self.was_paused = False

def main():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(32)  # Permitir múltiples canales simultáneos
    
    width, height = 1200, 720
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
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

    texture = load_texture(ctx, "placeholder.jpeg")
    texture.use(location=0)
    program['image'] = 0

    # ========== CONFIGURAR CAPAS DE AUDIO ==========
    audio_folder = "Audios"
    audio_files = []
    
    # Buscar archivos de audio
    if os.path.exists(audio_folder):
        audio_files = sorted([
            os.path.join(audio_folder, f) 
            for f in os.listdir(audio_folder) 
            if f.lower().endswith(('.mp3', '.wav', '.ogg'))
        ])
    
    # Crear capas de audio distribuidas uniformemente
    audio_layers = []
    if audio_files:
        num_audios = len(audio_files)
        print(f"\n🎵 Encontrados {num_audios} archivos de audio")
        
        for i, audio_file in enumerate(audio_files):
            # Distribuir entre 0.0 y 0.95 (dejar margen al final)
            activation_point = (i / max(num_audios - 1, 1)) * 0.95
            
            # Volumen decreciente para evitar saturación
            volume = 1.0 - (i / num_audios) * 0.3
            
            layer = AudioLayer(audio_file, activation_point, volume)
            if layer.load():
                audio_layers.append(layer)
                print(f"  ✓ {os.path.basename(audio_file)} → activa en {activation_point:.2f}")
    else:
        print(f"\n⚠️  No se encontraron archivos de audio en '{audio_folder}'")
    
    print(f"\n{'='*60}")
    print(f"🎮 CONTROLES:")
    print(f"  ESPACIO = Pausar/Reanudar")
    print(f"  R       = Resetear")
    print(f"  ESC     = Salir")
    print(f"{'='*60}\n")

    # Variables de tiempo y estado
    start_time = time.time()
    duration = 60.0
    isPaused = False
    resetting = False
    reset_duration = 3.0
    reset_start = 0.0
    reset_from = 0.0
    clock = pygame.time.Clock()
    
    button_pos = np.array([width - 80.0, height - 80.0], dtype='f4')
    button_size = np.array([60.0, 60.0], dtype='f4')

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                # Salir
                if event.key == K_ESCAPE:
                    running = False
                    
                # Reset completo
                elif event.key == K_r:
                    print("🔄 Reseteando...")
                    for layer in audio_layers:
                        layer.stop()
                    start_time = time.time()
                    resetting = False
                    isPaused = False
                    
                # Pausar/reanudar con transición
                elif event.key == K_SPACE:
                    if not isPaused and not resetting:
                        # Iniciar animación de pausa
                        resetting = True
                        reset_start = time.time()
                        reset_from = min(1.0, (time.time() - start_time) / duration)
                        print("⏸️  Pausando...")
                    elif isPaused:
                        # Reanudar
                        isPaused = False
                        start_time = time.time()
                        resetting = False
                        # Resetear audios para que empiecen sincronizados
                        for layer in audio_layers:
                            layer.stop()
                        print("▶️  Reanudando...")

        elapsed = time.time() - start_time    

        # Calcular progress
        if resetting:
            t = (time.time() - reset_start) / reset_duration
            if t >= 1.0:
                isPaused = True
                progress = 0.0
                resetting = False
            else:
                progress = reset_from * (1.0 - t)
        else:
            progress = min(1.0, elapsed / duration)

        # Actualizar todas las capas de audio
        for layer in audio_layers:
            layer.update(progress, isPaused)

        # Renderizar
        ctx.clear(0.0, 0.0, 0.0)
        
        program['progress'].value = progress
        program['time'].value = elapsed
        program['isPaused'].value = isPaused
        vao.render(moderngl.TRIANGLE_STRIP)

        ui_program['resolution'].value = (width, height)
        ui_program['buttonPos'].value = tuple(button_pos)
        ui_program['buttonSize'].value = tuple(button_size)
        ui_program['isPaused'].value = isPaused
        ui_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    for layer in audio_layers:
        layer.stop()
    pygame.quit()

if __name__ == "__main__":
    main()