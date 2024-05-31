#version 420 core

in vec2 uv;

out vec4 out_color;

uniform sampler2D tex;

void main() {
	out_color = textureLod(tex, uv, 0);
}
