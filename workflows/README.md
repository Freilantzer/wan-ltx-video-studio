# Workflows

This folder will hold versioned ComfyUI workflow templates and manifests.

Planned structure:

```text
workflows/
  wan/
    wan22-ti2v-5b-t2v.workflow_api.json
    wan22-ti2v-5b-i2v.workflow_api.json
    wan22-t2v-a14b.workflow_api.json
    wan22-i2v-a14b.workflow_api.json
  ltx/
    ltx23-t2v.workflow_api.json
    ltx23-i2v.workflow_api.json
  manifests/
    wan22-ti2v-5b-t2v.json
```

Workflow JSON files should stay small and versionable. Do not place model weights or generated outputs here.

