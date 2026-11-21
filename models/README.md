# Trained models

The weights are not in git (too big). They are attached to the GitHub release.

Upload: on the repo page -> Releases -> Draft a new release -> tag `v1.0` -> drag the files from `~/drone_ws/yolo/models/` in -> Publish.  (or with the GitHub CLI: `gh release create v1.0 ~/drone_ws/yolo/models/*.pt`)

Use: download them into `~/drone_ws/yolo/models/`.

| file | size | sha256 (first 16) |
|---|---|---|
| `best.pt` | 5.4 MB | `a8af1ebb29facd57` |
| `bridge5_y11s.pt` | 19.2 MB | `8848bda3f70ddefb` |
| `bridge5_y11s_analog.pt` | 19.2 MB | `ccaf89bc8cfaa64e` |
| `bridge5_y11s_droneview.pt` | 19.2 MB | `09c2d1366397f8f3` |
| `bridge5_y11s_e6.pt` | 19.2 MB | `65b7f5a1e48af853` |
| `bridge5_y26s.pt` | 20.3 MB | `17d82898ef58bcfb` |
| `codebrim_y11s_640.pt` | 19.2 MB | `df3390feb715d970` |
| `dacl19_y11s.pt` | 19.2 MB | `0b3a5e2174e6b800` |
| `rdd4_y11s.pt` | 19.2 MB | `2e721f2916bb6b8d` |
| `rdd4_y11s_e6.pt` | 19.2 MB | `4d20fcadcee17401` |

See docs/04_models.md for what each one is.
