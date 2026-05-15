---
id: 160cd7cf05
question: No Space Left on Device - OSError[Errno 28]
sort_order: 19
---

You do not have enough disk space to install the requirements. Here are some solutions:

- **Increase EBS Volume on AWS:** Follow [this guide](https://n2ws.com/blog/how-to-guides/how-to-increase-the-size-of-an-aws-ebs-cloud-volume-attached-to-a-linux-machine#:~:text=First%2C%20go%20to%20your%20volume,your%20requirements%20necessitate%20this%20step) to increase the base EBS volume.

- **Add an External Disk on AWS:** Add and configure an external disk to your instance, then configure conda installation to happen on this external disk.

- **Add Persistent Disk on GCP:**
  1. Add another disk to your VM and follow [this guide](https://cloud.google.com/compute/docs/disks/add-persistent-disk) to mount the disk.
  2. Confirm the mount by running the following command in the bash shell:
     
    ```bash
    df -H
    ```
  3. Delete Anaconda and use Miniconda instead. Download Miniconda on the additional disk that you mounted.
  4. During the Miniconda installation, enter the path to the extra disk instead of the default disk, so that conda is installed on the extra disk.

</ANSWER>