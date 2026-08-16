# Wiping the repo and pushing this structure

This nukes the entire commit history. There is no undo once you force-push,
so make a backup branch on GitHub first if you have any doubt.

## 0. Backup (optional but cheap)

    git checkout main
    git branch backup-old-history
    git push origin backup-old-history

## 1. Point a local folder at your repo

    cd path/to/interspeech-repo        # existing clone
    # or: git clone <YOUR_REPO_URL> && cd <repo>

## 2. Delete everything tracked

    git rm -r --cached .
    # then delete the working files too, except .git
    # (Windows PowerShell)
    Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force

## 3. Copy the new files in

Copy the contents of the `github_repo` folder into the repo root, so you get:

    README.md  LICENSE  .gitignore  requirements.txt
    data/  scripts/  cs_wer/  configs/

## 4. Create an orphan branch with no history

    git checkout --orphan clean-main
    git add -A
    git commit -m "Dynamic Block-Online Streaming ASR for Agglutinative Code-Switching (Interspeech 2026)"

## 5. Replace main and force-push

    git branch -D main
    git branch -m main
    git push -f origin main

## 6. Clean up remote branches you no longer want

    git push origin --delete <old-branch>        # repeat per branch

## 7. Verify

Open the repo on GitHub. Commit count should read 1. If old branches or tags
still show, delete them under Settings or with:

    git push origin --delete <tag>

## Note on GitHub's cache

Force-pushing removes the branch pointer, but old commits can stay reachable
through their SHA for a while. If the old content was sensitive, the only real
fix is deleting the repo and creating a new one with the same name.
