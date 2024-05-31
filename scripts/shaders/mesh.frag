#version 420 core

in vec2 uv_;

layout(location=0) out vec4 color;

void main() {
    color = vec4(uv_, 0.0, 1.0);
}
