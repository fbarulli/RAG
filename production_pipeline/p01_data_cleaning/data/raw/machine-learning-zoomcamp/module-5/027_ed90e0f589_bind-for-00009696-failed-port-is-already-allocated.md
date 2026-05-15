---
id: ed90e0f589
question: 'Bind for 0.0.0.0:9696 failed: port is already allocated'
sort_order: 27
---

I was getting the following error when I rebuilt the Docker image, although the port was not allocated, and it was working fine.

Error message:

```
Error response from daemon: driver failed programming external connectivity on endpoint beautiful_tharp (875be95c7027cebb853a62fc4463d46e23df99e0175be73641269c3d180f7796): Bind for 0.0.0.0:9696 failed: port is already allocated.
```



The issue can be resolved by running the following command:

```bash
docker kill $(docker ps -q)
```

For more information, refer to the [GitHub issue on Docker for Windows](https://github.com/docker/for-win/issues/2722).