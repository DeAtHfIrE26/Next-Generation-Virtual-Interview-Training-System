# Interviewer avatar

`interviewer.glb` is the MPFB character from the TalkingHead project, optimised for the web.

| | |
|---|---|
| Source | https://raw.githubusercontent.com/met4citizen/TalkingHead/b3e277b3b46f88e557bf28a2c5612a5b04e075c3/avatars/mpfb.glb |
| Source SHA-256 | `63c645a2a863b9972e9a9c2ed576a1de4c390b8475508e1473e69c87a3ee299c` (36,815,920 bytes) |
| Licence | **CC0 1.0** (TalkingHead README: "mpfb.glb ... CC0"; created with Blender + MPFB, whose assets are CC0) |
| Rig and shapes | Mixamo-compatible armature (67 joints), 52 ARKit blendshapes + 14 Oculus visemes |
| Optimisation | `scripts/optimize-avatar.sh`: dedup, prune, quantize (KHR_mesh_quantization), textures to 2048 px WebP (EXT_texture_webp), meshopt (EXT_meshopt_compression). 36.8 MB to 4.0 MB (2.9 MB gzipped). Rig and all 66 morph targets verified after optimisation. |

The other sample avatars in the TalkingHead repository are non-commercial (Ready Player Me, Avaturn,
AvatarSDK, VRoid) and are not used. See docs/DECISIONS.md D4.
