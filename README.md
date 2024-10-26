# toy-pathtracer
This is the playground where I experiment and learn about path tracing 🙃

<img src="https://github.com/kadir014/toy-pathtracer/blob/cafb575dd2b74f1a6565613e4e9e4ae5eb3187fc/thumbnail.png" width=800>



# Features
- [x] Progressive rendering.
- [x] Basic camera setup & movement.
- [x] Final image adjustments (exposure, tonemapping).
- [x] Anti-aliasing (jittered sampling?).
- [x] Basic materials:
  - [x] Diffuse
  - [x] Emissive
  - [x] Specular (reflective)
  - [ ] Refractive
- [x] Collision shapes:
  - [x] Sphere
  - [x] Triangle
- [x] Russian roulette path termination.
- [ ] A more physically accurate material model (perhaps Disney PBR?).
- [ ] Loading complex geometry & meshes.
- [ ] BVH tree for spatial acceleration.
- [ ] Texture mapping.
- [ ] Depth of field.
- [ ] Volumetric rendering.
- [ ] Denoising.
- [ ] Learn explicit light sampling, MIS and such.



# Running
You need Python 3.10+. After cloning the repo, install required packages:
```sh
$ pip install -r requirements.txt
```
And then just run `main.py`
```sh
$ python main.py
```


# Resources & References
- P. Shirley, T. D. Black, S. Hollasch, [Ray Tracing in One Weekend Series](https://raytracing.github.io/)
- Demofox, [Casual Shadertoy Path Tracing Blog](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/)
- Sebastian Lague, [Coding Adventure: Ray Tracing](https://www.youtube.com/watch?v=Qz0KTGYJtUk)
- scratchapixel.com, [Ray tracing articles](https://www.scratchapixel.com/)



# License
[MIT](LICENSE) © Kadir Aksoy
