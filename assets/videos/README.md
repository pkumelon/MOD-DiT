# Video Comparison Assets

Place the qualitative comparison clips referenced by the README in this directory.

## Expected files

| File | Content |
|:---|:---|
| `hunyuan_full.gif` / `hunyuan_full.mp4` | HunyuanVideo, full attention |
| `hunyuan_moddit.gif` / `hunyuan_moddit.mp4` | HunyuanVideo, MOD-DiT |
| `wan_full.gif` / `wan_full.mp4` | Wan2.1, full attention |
| `wan_moddit.gif` / `wan_moddit.mp4` | Wan2.1, MOD-DiT |
| `cogvideox_full.gif` / `cogvideox_full.mp4` | CogVideoX-v1.5, full attention |
| `cogvideox_moddit.gif` / `cogvideox_moddit.mp4` | CogVideoX-v1.5, MOD-DiT |

## Guidelines

- Each comparison pair must use the **same prompt and seed**, differing only in `--sparse_type`.
- GitHub renders GIFs inline but does not autoplay local MP4 files inside README tables, so a GIF is required for the inline view.
- Keep each GIF under roughly 10 MB to avoid slow page loads. If a clip is too large, lower the frame rate or width.
- Committing large binaries directly bloats the repository. Consider [Git LFS](https://git-lfs.com/) or hosting the MP4 files externally and linking to them instead.

## Converting MP4 to GIF

```bash
ffmpeg -i input.mp4 \
  -vf "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
  -loop 0 output.gif
```

Lower `fps` or `scale` to further reduce file size.
