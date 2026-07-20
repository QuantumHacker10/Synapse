# Promotion & community

Tips for sharing Synapse OMNIA with a wider audience.

## Demos and visuals

High-impact content for GitHub, the landing page, and social posts:

| Format | Suggestion |
|---|---|
| **GIF** | 15–30 s loop: Studio viewport, evolution mutating a neural SDF, living-law swap |
| **Video** | 2–5 min walkthrough: load `demo.synapse`, pause, rewrite a law, LLM lighting hint |
| **Screenshots** | Before/after L-DNN GI, G-DNN LOD levels, physics joints/vehicles |

Store assets under `docs/media/` (create when ready) and link from README or GitHub Releases notes.

### Recording tips

- Use `--scene samples/demo.synapse` for a reproducible starting point
- Pause (Space) before camera moves for cleaner captures
- Mention Vulkan + .NET 10 prerequisites in video descriptions

## Where to share

| Channel | Angle |
|---|---|
| [r/CSharp](https://reddit.com/r/csharp) | .NET 10, Avalonia Studio, CI/CD, simulation architecture |
| [Hacker News](https://news.ycombinator.com/submit) | “3D simulation engine with hot-reloadable physics laws and neural SDF/GI” |
| Graphics / engine forums | G-DNN ray marching, L-DNN GI, Vulkan RHI |
| Academic networks | Living laws, NEAT-G shape evolution, digital twins |

Always link to the official repository and Releases page — not unofficial mirrors.

## GitHub presence

- Use **English** for README, issues, and PRs to reach an international audience
- Tag releases with **Semantic Versioning** (`v1.1.0`, `v1.2.0`, …)
- Enable **Discussions** for architecture Q&A
- Add **CI badges** (build, analysis, coverage) to README — see current badges

## License reminder

Synapse OMNIA is under the **MIT License**. You may share demos, fork the repo, and
redistribute modified builds — include the LICENSE file. See [LICENSE-OPTIONS.md](LICENSE-OPTIONS.md).
