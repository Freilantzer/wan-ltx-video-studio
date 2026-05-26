# GitHub Setup

## Local Repo

Local folder:

```text
D:\VIDEO_GENS\wan-ltx-video-studio
```

The repository has been initialized locally with Git. The default branch should be renamed to `main` before the first commit if it is not already.

## Needed From You

To publish this to GitHub, choose:

- Repo name: recommended `wan-ltx-video-studio`
- Visibility: `private` while planning, unless you already want it public
- License: recommended `MIT` for the app code, but we must keep model licenses separate
- Description: `Local-first WAN 2.2 video generation studio with future LTX support`

## Auth Options

### Option A: GitHub CLI

Install GitHub CLI, then run:

```powershell
gh auth login
```

After that I can create and push:

```powershell
gh repo create wan-ltx-video-studio --private --source . --remote origin --push
```

### Option B: Existing Remote URL

Create an empty repo on GitHub, then give me the HTTPS or SSH URL. I can add it:

```powershell
git remote add origin <repo-url>
git push -u origin main
```

## Important Repo Rules

Do not commit:

- Model weights
- Generated videos/images/audio
- ComfyUI installs
- API tokens
- Local machine paths containing secrets

The `.gitignore` is configured for those already.

