#version 130

in vec3 position;
in vec2 uv;

out vec2 uv_;

void main() {
    gl_Position = gl_ModelViewProjectionMatrix * vec4(position, 1.0);
    uv_ = uv;
}
