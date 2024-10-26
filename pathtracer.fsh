/*
    Toy Path Tracer Project
    https://github.com/kadir014/toy-pathtracer
*/

#version 330

in vec2 v_uv;
out vec4 f_color;

uniform sampler2D s_sky;
uniform sampler2D s_prev;
uniform vec2 u_viewport;
uniform uint u_frame;
uniform uint u_true_frame;
uniform bool u_roulette;


#define PI  3.141592653589793238462643383279
#define TAU 6.283185307179586476925286766559
#define MAXVAL 10000.0
#define EPSILON 0.001
#define NORMAL_NUDGE EPSILON

#define MAX_DEPTH 5
#define SAMPLES 2


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


struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Material {
    vec3 color;
    vec3 emissive;
    float specular_percentage;
    vec3 specular_color;
    float roughness;
};

struct HitInfo {
    bool hit;
    vec3 point;
    vec3 normal;
    Material material;
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

struct Triangle {
    vec3 v0;
    vec3 v1;
    vec3 v2;
    vec3 normal;
    Material material;
};

struct Quad {
    vec3 v0;
    vec3 v1;
    vec3 v2;
    vec3 v3;
    Material material;
};

struct Camera {
    vec3 position;
    vec3 center;
    vec3 u;
    vec3 v;
};

uniform Camera u_camera;


/*
    Sphere x Ray intersection function by Inigo Quilez
    https://iquilezles.org/articles/intersectors/
*/
HitInfo sphere_x_ray(Sphere sphere, Ray ray) {
    HitInfo empty_hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        Material(vec3(0.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
    );

    vec3 oc = ray.origin - sphere.center;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float h = b * b - c;

    // No intersection
    if (h < 0.0) return empty_hitinfo;
    h = sqrt(h);

    vec2 t = vec2(-b-h, -b+h);

    // Ray does NOT intersect the sphere
    if (t.y < 0.0 ) return empty_hitinfo;

    float d = 0.0;
    if (t.x < 0.0) d = t.y; // Ray origin inside the sphere, t.y is intersection distance
    else d = t.x; // Ray origin outside the sphere, t.x is intersection distance

    vec3 intersection = ray.origin + ray.dir * d;
    vec3 normal = normalize(intersection - sphere.center);

    return HitInfo(true, intersection, normal, sphere.material);
}

/*
    Triangle x Ray intersection function by Inigo Quilez
    https://iquilezles.org/articles/intersectors/
*/
HitInfo triangle_x_ray(Triangle triangle, Ray ray) {
    HitInfo empty_hitinfo = HitInfo(
        false,
        vec3(0.0),
        vec3(0.0),
        Material(vec3(0.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
    );

    vec3 v1v0 = triangle.v1 - triangle.v0;
    vec3 v2v0 = triangle.v2 - triangle.v0;
    vec3 rov0 = ray.origin - triangle.v0;
    vec3 n = cross(v1v0, v2v0);
    vec3 q = cross(rov0, ray.dir);
    float d = 1.0 / dot(ray.dir, n);
    float u = d * dot(-q, v2v0);
    float v = d * dot(q, v1v0);
    float t = d * dot(-n, rov0);

    if (u < 0.0 || v < 0.0 || (u + v) > 1.0 ) return empty_hitinfo;
    if (t <= 0.0) return empty_hitinfo;

    vec3 intersection = ray.origin + t * ray.dir;
    vec3 normal = triangle.normal;
    if (dot(normal, ray.dir) > 0.0) normal = -normal;

    return HitInfo(true, intersection, normal, triangle.material);
}

/*
    https://en.wikipedia.org/wiki/UV_mapping#Finding_UV_on_a_sphere
*/
vec2 uv_project_sphere(vec3 pos) {
    float u = 0.5 + atan(pos.z, pos.x) / TAU;
    float v = 0.5 + asin(pos.y) / PI;

    return vec2(u, v);
}


/*
    PCG PRNG from https://github.com/riccardoscalco/glsl-pcg-prng/blob/main/index.glsl
*/

uint pcg(uint v) {
	uint state = v * uint(747796405) + uint(2891336453);
	uint word = ((state >> ((state >> uint(28)) + uint(4))) ^ state) * uint(277803737);
	return (word >> uint(22)) ^ word;
}

float prng(inout uint prng_state) {
    uint new_state = pcg(prng_state);
    prng_state = new_state;
	return float(new_state) / 4294967296.0;
}

vec3 random_in_unit_sphere(inout uint state) {
    float z = prng(state) * 2.0 - 1.0;
    float a = prng(state) * TAU;
    float r = sqrt(1.0 - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return vec3(x, y, z);
}

Ray scatter(Ray ray, HitInfo hitinfo, inout float specular, inout uint prng_state) {
    vec3 new_pos = hitinfo.point + hitinfo.normal * NORMAL_NUDGE;

    specular = (prng(prng_state) < hitinfo.material.specular_percentage) ? 1.0 : 0.0;

    vec3 diffuse_ray_dir = normalize(hitinfo.normal + random_in_unit_sphere(prng_state));
    vec3 specular_ray_dir = reflect(ray.dir, hitinfo.normal);
    specular_ray_dir = normalize(mix(specular_ray_dir, diffuse_ray_dir, hitinfo.material.roughness * hitinfo.material.roughness));

    vec3 new_dir = mix(diffuse_ray_dir, specular_ray_dir, specular);

    return Ray(new_pos, new_dir);
}

/*
    Generate ray from camera position to screen position.
*/
Ray generate_ray(vec2 pos) {
    vec3 screen_world = (u_camera.center + u_camera.u * pos.x) + u_camera.v * pos.y;

    return Ray(
        u_camera.position,
        normalize(screen_world - u_camera.position)
    );
}

/*
    Cast the ray into the scene and gather collided objects.
*/
HitInfo cast_ray(Ray ray, Sphere[6] spheres, Triangle[22] tris, int skip_i) {
    float min_depth = MAXVAL;
    HitInfo min_hitinfo = HitInfo(false, vec3(0.0), vec3(0.0), Material(vec3(0.0), vec3(0.0), 0, vec3(0.0), 0.0));
    
    for (int i = 0; i < 6; i++) {
        if (i == skip_i) {continue;}

        HitInfo hitinfo = sphere_x_ray(spheres[i], ray);

        if (hitinfo.hit) {
            float dist = distance(hitinfo.point, ray.origin);

            if (dist < min_depth) {
                min_depth = dist;
                min_hitinfo = hitinfo;
            }
        }
    }

    for (int i = 0; i < 22; i++) {
        if (i == skip_i) {continue;}

        HitInfo hitinfo = triangle_x_ray(tris[i], ray);

        if (hitinfo.hit) {
            float dist = distance(hitinfo.point, ray.origin);

            if (dist < min_depth) {
                min_depth = dist;
                min_hitinfo = hitinfo;
            }
        }
    }

    return min_hitinfo;
}

/*
    Path Trace!
*/
vec3 trace(Ray ray, Sphere[6] spheres, Triangle[22] tris, inout uint prng_state) {
    vec3 ret_color = vec3(0.0);
    vec3 ray_color = vec3(1.0);
    Ray nray = Ray(ray.origin, ray.dir);

    for (int i = 0; i < MAX_DEPTH; i++) {
        
        HitInfo hitinfo = cast_ray(nray, spheres, tris, -1);

        // Ray did not hit anything, sample sky
        if (!hitinfo.hit) {
            vec3 sky_color = srgb_to_rgb_approx(texture(s_sky, uv_project_sphere(nray.dir)).rgb);
            ret_color += sky_color * ray_color;
            break;
        }

        float specular = 0.0;
        nray = scatter(nray, hitinfo, specular, prng_state);

        ret_color += hitinfo.material.emissive * ray_color;
        ray_color *= mix(hitinfo.material.color, hitinfo.material.specular_color, specular);

        /*
            Russian Roulette:
            As the throughput gets smaller, the ray is more likely to get terminated early.
            Survivors have their value boosted to make up for fewer samples being in the average.
        */
        if (u_roulette) {
            float roulette = max(ray_color.r, max(ray_color.g, ray_color.b));
            if (prng(prng_state) > roulette) {
                break;
            }
        
            // Add the energy we 'lose' by randomly terminating paths
            ray_color *= 1.0 / roulette;
        }
    }

    return ret_color;
}


void main() {
    vec2 p = v_uv * u_viewport;
    // Initialize a random state based on frag coords and frame no multiplied by some primes
    uint prng_state = uint(p.x) * uint(2791) + uint(p.y) * uint(7129) + u_true_frame * uint(23857);

    // Generate scene
    Sphere[6] spheres = Sphere[](
        Sphere(
            vec3(-10.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 0.0)
        ),
        Sphere(
            vec3(-3.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 0.333)
        ),
        Sphere(
            vec3(3.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 0.667)
        ),
        Sphere(
            vec3(10.5, 0.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(0.3, 0.1, 0.8), 1.0)
        ),

        Sphere(
            vec3(0.0, 13.0, 35.0),
            3.0,
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        Sphere(
            vec3(0.0, 14.0, 25.0),
            1.0,
            Material(vec3(0.0), vec3(1.0) * 2.0, 0.0, vec3(0.0), 0.0)
        )
    );

    Quad[11] quads = Quad[](
        // Floor
        Quad(
            vec3(-15.0, -15.0, 45.0),
            vec3( 15.0, -15.0, 45.0),
            vec3( 15.0, -15.0, 15.0),
            vec3(-15.0, -15.0, 15.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Ceiling
        Quad(
            vec3(-15.0, 15.0, 45.0),
            vec3( 15.0, 15.0, 45.0),
            vec3( 15.0, 15.0, 15.0),
            vec3(-15.0, 15.0, 15.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Back wall
        Quad(
            vec3(-15.0, -15.0, 45.0),
            vec3( 15.0, -15.0, 45.0),
            vec3( 15.0,  15.0, 45.0),
            vec3(-15.0,  15.0, 45.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Front wall
        Quad(
            vec3(-15.0, -15.0, 15.0),
            vec3( 15.0, -15.0, 15.0),
            vec3( 15.0,  15.0, 15.0),
            vec3(-15.0,  15.0, 15.0),
            Material(vec3(1.0), vec3(0.0), 1.0, vec3(1.0, 0.5, 0.5), 0.35)
        ),

        // Left wall
        Quad(
            vec3(15.0, -15.0, 45.0),
            vec3(15.0, -15.0, 15.0),
            vec3(15.0,  15.0, 15.0),
            vec3(15.0,  15.0, 45.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Right wall
        Quad(
            vec3(-15.0, -15.0, 45.0),
            vec3(-15.0, -15.0, 15.0),
            vec3(-15.0,  15.0, 15.0),
            vec3(-15.0,  15.0, 45.0),
            Material(vec3(1.0), vec3(0.0), 0.0, vec3(0.0), 0.0)
        ),

        // Light
        Quad(
            vec3(-13.0, -15.0, 44.9),
            vec3( -11.0, -15.0, 44.9),
            vec3( -11.0,  15.0, 44.9),
            vec3(-13.0,  15.0, 44.9),
            Material(vec3(0.0), vec3(1.0, 0.0, 0.0) * 4.0, 0.0, vec3(0.0), 0.0)
        ),

        Quad(
            vec3(-7.0, -15.0,  44.9),
            vec3( -5.0, -15.0, 44.9),
            vec3( -5.0,  15.0, 44.9),
            vec3(-7.0,  15.0,  44.9),
            Material(vec3(0.0), vec3(1.0, 0.5, 0.0) * 4.0, 0.0, vec3(0.0), 0.0)
        ),

        Quad(
            vec3(-1.0, -15.0, 44.9),
            vec3( 1.0, -15.0, 44.9),
            vec3( 1.0,  15.0, 44.9),
            vec3(-1.0,  15.0, 44.9),
            Material(vec3(0.0), vec3(0.0, 1.0, 0.0) * 4.0, 0.0, vec3(0.0), 0.0)
        ),

        Quad(
            vec3(5.0, -15.0, 44.9),
            vec3(7.0, -15.0, 44.9),
            vec3(7.0,  15.0, 44.9),
            vec3(5.0,  15.0, 44.9),
            Material(vec3(1.0), vec3(0.0, 0.5, 1.0) * 4.0, 0.0, vec3(1.0), 0.2)
        ),

        Quad(
            vec3(13.0, -15.0, 44.9),
            vec3(11.0, -15.0, 44.9),
            vec3(11.0,  15.0, 44.9),
            vec3(13.0,  15.0, 44.9),
            Material(vec3(1.0), vec3(0.5, 0.0, 1.0) * 4.0, 0.0, vec3(1.0), 0.2)
        )
    );

    // Calculate triangles from quads
    Triangle[22] tris;
    int j = 0;
    for (int i = 0; i < 22; i += 2) {
        Triangle tri0 = Triangle(quads[j].v0, quads[j].v1, quads[j].v2, vec3(0.0), quads[j].material);
        Triangle tri1 = Triangle(quads[j].v2, quads[j].v3, quads[j].v0, vec3(0.0), quads[j].material);
        tri0.normal = normalize(cross(tri0.v1 - tri0.v0, tri0.v2 - tri0.v0));
        tri1.normal = normalize(cross(tri1.v1 - tri1.v0, tri1.v2 - tri1.v0));
        tris[i] = tri0;
        tris[i + 1] = tri1;
        j++;
    }

    vec3 final_color = vec3(0.0);

    for (int i = 0; i < SAMPLES; i++) {
        // Anti-aliasing
        float rx = (prng(prng_state) * 2.0 - 1.0) / 790.0;
        float ry = (prng(prng_state) * 2.0 - 1.0) / 790.0;
        vec2 pos = v_uv * 2.0 - 1.0 + vec2(rx, ry);

        Ray ray = generate_ray(pos);

        vec3 ray_color = trace(ray, spheres, tris, prng_state);
        
        final_color += ray_color / float(SAMPLES);
    }

    // Accumulate and average the previous and current frame for progressive rendering
    vec3 prev_color = texture(s_prev, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
    float weight = 1.0 / float(u_frame + uint(1));
    vec3 color = mix(prev_color, final_color, weight);

    f_color = vec4(color, 1.0);
}