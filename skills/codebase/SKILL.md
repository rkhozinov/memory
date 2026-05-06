---
name: codebase
description: Index, search, and view the codebase structural map from persistent memory. Use when navigating repository structure, finding layers/modules/services, or refreshing the codebase index.
argument-hint: "[index [--refresh] | search <query> | (no args = show all)]"
allowed-tools: Bash
---

# /codebase

Set `PROJECT=$(basename "$(pwd)")` before commands.

## No arguments (`/codebase`) — Show full map

```bash
memory search --tags "project:$PROJECT,scope:codebase-map" -n 50
```

Present results grouped by category: Terraform Layers, Terraform Modules, K8s Apps, K8s Services, Relationships.

## Search (`/codebase search <query>`)

```bash
memory search "<query>" --tags "project:$PROJECT,scope:codebase-map" -n 10
```

## Index (`/codebase index [--refresh]`)

Scan the repository and store structural metadata into memory.

### Steps

1. Set variables:
   ```bash
   PROJECT=$(basename "$(pwd)")
   GIT_SHA=$(git rev-parse --short HEAD)
   INDEXED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
   MEMORY="$HOME/.claude/tools/memory/.venv/bin/memory"
   ```

2. If `--refresh` flag: delete old entries first:
   ```bash
   memory delete --tags "project:$PROJECT,scope:codebase-map" --yes
   ```

3. Scan and build entries. For each category, collect info via filesystem exploration:

   **Terraform layers** — directories under `terraform/` containing `*.tf` files (excluding `modules/`):
   ```bash
   find terraform/ -name '*.tf' -not -path '*/modules/*' -exec dirname {} \; | sort -u
   ```
   For each layer dir, list the `.tf` files present. Store as:
   ```
   [Reference] TF layer: <relative-path> — Files: backend.tf, main.tf, ...
   ```
   Tags: `project:$PROJECT,scope:codebase-map,layer:<relative-path>`

   **Terraform modules** — directories under `terraform/modules/`:
   ```bash
   find terraform/modules/ -name '*.tf' -exec dirname {} \; | sort -u
   ```
   For each module dir, list the `.tf` files. Store as:
   ```
   [Reference] TF module: <cloud>/<name> — Files: main.tf, variables.tf, ...
   ```
   Tags: `project:$PROJECT,scope:codebase-map,module:<cloud>/<name>`

   **Kubernetes apps** — directories under `kubernetes/apps/`:
   ```bash
   ls -d kubernetes/apps/*/
   ```
   For each app, list base/ and overlays/. Store as:
   ```
   [Reference] K8s app: <name> — base + overlays: <overlay-list>
   ```
   Tags: `project:$PROJECT,scope:codebase-map,k8s-app:<name>`

   **Kubernetes cluster-services** — directories under `kubernetes/cluster-services/`:
   ```bash
   ls -d kubernetes/cluster-services/*/
   ```
   Store as:
   ```
   [Reference] K8s service: <name> — <key-files>
   ```
   Tags: `project:$PROJECT,scope:codebase-map,k8s-svc:<name>`

   **Relationships** — parse `data.tf` files for `terraform_remote_state` blocks:
   ```bash
   grep -r 'terraform_remote_state' terraform/ --include='data.tf' -l
   ```
   For each `data.tf`, extract the remote state references and the layer that reads them. Store as:
   ```
   [Pattern] <layer> reads remote state from <source-layer> (<key-outputs>)
   ```
   Tags: `project:$PROJECT,scope:codebase-map,relationship:<layer>-><source>`

4. Store all entries using `memory store` with `--dedup 0.85` for each entry. Include metadata:
   ```bash
   memory store "<content>" \
     --tags "<tags>" \
     --type reference \
     --dedup 0.85 \
     --metadata '{"path":"<path>","git_sha":"'"$GIT_SHA"'","indexed_at":"'"$INDEXED_AT"'"}'
   ```

5. Report summary: number of entries stored per category.
