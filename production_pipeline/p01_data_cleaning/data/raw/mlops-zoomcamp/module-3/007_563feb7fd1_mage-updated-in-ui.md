---
id: 563feb7fd1
question: Mage updated in UI
sort_order: 7
---

When you see the mage version change in the UI after you’ve started the container, and you want to update, follow these steps. Read the release notes first to see if there’s a fix that affected your work and would benefit from an update.

If you want to remain in the previous version, it's also fine unless the fixes were specifically for our zoomcamp coursework (check the repository for any new instructions or PRs added).

1. Close the browser page.
2. In the terminal console, bring down the container:
   ```bash
   docker compose down
   ```
3. Rebuild the container with the new mage image:
   ```bash
   docker compose build --no-cache
   ```
4. Verify that you see:
   ```bash
   [magic-platform 1/4] FROM docker.io/mageai/mageai:alpha
   ```
   This means that the container is being rebuilt with a new version.

5. If the image is not updated, press `ctrl+c` to cancel the process and pull the image manually:
   ```bash
   docker pull mageai/mageai:alpha
   ```
   Then rebuild.

6. Restart the docker container as before:
   ```bash
   ./scripts/start.sh
   ```

Note: This is the same sequence of steps if you want to switch to the latest tagged image instead of using the alpha image.

**What does alpha and latest mean?**

- **Latest** is the fully released version ready for production use, and it has gone through verification, testing, QA, and whatever else the release cycle entails.

- **Alpha** is the potentially buggy version with fresh new fixes and newly added features; but not yet put through the full beta test (if there’s one), integration testing, and other QA steps. Expect issues to occur.