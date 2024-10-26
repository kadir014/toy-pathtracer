/*
    Toy Path Tracer Project
    https://github.com/kadir014/toy-pathtracer
*/

#version 330

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_display;
uniform float u_exposure;


/*
    sRGB <=> Linear RGB conversion functions
    https://github.com/tobspr/GLSL-Color-Spaces/blob/master/ColorSpaces.inc.glsl
*/

const float SRGB_GAMMA = 1.0 / 2.2;
const float SRGB_INVERSE_GAMMA = 2.2;

vec3 rgb_to_srgb_approx(vec3 rgb) {
    return pow(rgb, vec3(SRGB_GAMMA));
}

vec3 srgb_to_rgb_approx(vec3 srgb) {
    return pow(srgb, vec3(SRGB_INVERSE_GAMMA));
}

/*
    ACES filmic tone mapping curve
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
*/
vec3 aces_filmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0, 1.0);
}


void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    vec3 color = texture(s_display, uv).rgb;

    // Apply exposure -> Tonemap -> to sRGB
    color *= u_exposure;
    color = aces_filmic(color);
    color = rgb_to_srgb_approx(color);

    f_color = vec4(color, 1.0);
}