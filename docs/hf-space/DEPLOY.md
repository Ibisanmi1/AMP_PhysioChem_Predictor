# Hugging Face Space — deploy and push

**Space:** [Ibisanmi1/AMP_PhysioChemical_Predictor](https://huggingface.co/spaces/Ibisanmi1/AMP_PhysioChemical_Predictor)  
**App entry:** `app.py` (declared in `README.md` YAML frontmatter)

---

## 1. Install the Hugging Face CLI

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Ensure `hf` is on your `PATH` (often `~/.local/bin`).

---

## 2. Set your token (one-time per machine)

Create a token with **write** access to Spaces:  
[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Option A — script (recommended)**

```bash
cd /path/to/AMP_PhysioChem_Predictor
export HF_TOKEN=hf_your_token_here
chmod +x scripts/hf_set_token.sh scripts/hf_push_space.sh
./scripts/hf_set_token.sh
```

**Option B — interactive**

```bash
hf auth login
# paste token when prompted; add --add-to-git-credential for Git pushes to hf.co
```

**Option C — environment only (CI)**

```bash
export HF_TOKEN=hf_...
```

Many tools read `HF_TOKEN` automatically; Git still needs either `--add-to-git-credential` (see Option A) or a credential helper.

---

## 3. Clone the Space (optional, empty checkout)

When Git asks for a **password**, use the **token** (not your Hugging Face account password):

```bash
git clone https://huggingface.co/spaces/Ibisanmi1/AMP_PhysioChemical_Predictor
```

Or download files without Git:

```bash
hf download Ibisanmi1/AMP_PhysioChemical_Predictor --repo-type=space
```

---

## 4. Push this repository to the Space remote

From your **development clone** (this GitHub repo), add the Space as a second remote and push:

```bash
cd /path/to/AMP_PhysioChem_Predictor
./scripts/hf_push_space.sh
```

That pushes your current branch to `main` on the Space. To push another branch:

```bash
./scripts/hf_push_space.sh my-feature-branch
```

If Git prompts for credentials after `hf auth login --add-to-git-credential`, use:

- **Username:** your Hugging Face username  
- **Password:** the same `hf_...` token  

---

## 5. Space build notes

- **Port:** Spaces set `PORT`; `app.py` already uses it.
- **Errors in UI:** On Space, `show_error` defaults to **off** unless you set `GRADIO_SHOW_ERROR=true` in Space **Settings → Repository secrets → Variables** (or Variables UI).
- **Checkpoints:** Large `.pt` files may exceed Git LFS / Space limits if committed. Prefer **Hub model repo** + runtime `huggingface_hub` download, or Space **persistent storage**, per your hosting plan.
- **RDKit / build time:** First build can be slow; if the Space fails on RDKit, consider a **Docker Space** with conda-forge RDKit (advanced).

---

## 6. Sync GitHub → Space (alternative)

In the Space **Settings**, you can connect a **GitHub** repository so pushes to GitHub trigger rebuilds, instead of using `git push` to `huggingface.co`. Use one primary workflow to avoid conflicting histories.
