---
id: f884543e2c
question: Installing md5sum on Macos
sort_order: 29
---

To install `md5sum` on macOS, use the following command:

```bash
brew install md5sha1sum
```

Then run the command to check the hash for a file to see if it matches the provided hash:

```bash
md5sum model1.bin dv.bin
```