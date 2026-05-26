# GitHub Setup

## Local Repo

Local folder:

```text
D:\VIDEO_GENS\wan-ltx-video-studio
```

The repository has been initialized locally with Git and pushed to GitHub.

Remote:

```text
git@github.com:Freilantzer/wan-ltx-video-studio.git
```

Current setup:

- Repo name: recommended `wan-ltx-video-studio`
- Owner: `Freilantzer`
- Visibility: private
- Default branch: `main`
- License: not selected yet

## Auth

SSH auth is configured for this repository.

Useful commands:

```powershell
git status --short --branch
git pull --ff-only
git push
```

## Future Choices

- License: recommended MIT for the app code if this becomes public, while keeping model licenses separate.
- Description: `Local-first WAN 2.2 video generation studio with future LTX support`

## Important Repo Rules

Do not commit:

- Model weights
- Generated videos/images/audio
- ComfyUI installs
- API tokens
- Local machine paths containing secrets

The `.gitignore` is configured for those already.
