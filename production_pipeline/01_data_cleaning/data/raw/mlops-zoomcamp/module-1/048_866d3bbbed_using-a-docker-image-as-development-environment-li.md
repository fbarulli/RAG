---
id: 866d3bbbed
question: Using a docker image as development environment (Linux)
sort_order: 48
---

If you don’t want to install Anaconda locally and prefer not to use Codespace or a VPS, you can create and run a Docker image locally.

For this, use the following `Dockerfile`:

```dockerfile
FROM docker.io/bitnami/minideb:bookworm

RUN install_packages wget ca-certificates vim less silversearcher-ag

# Uncomment the `COPY` and comment the `RUN` line if you have downloaded anaconda manually
# I did this to save bandwidth when experimenting with the image creation

RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh && bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /opt/anaconda3

#COPY Anaconda3-2022.05-Linux-x86_64.sh /tmp/Anaconda3-2022.05-Linux-x86_64.sh

RUN bash /tmp/Anaconda3-2022.05-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    rm /tmp/Anaconda3-2022.05-Linux-x86_64.sh

ENV PATH="/opt/anaconda3/bin:$PATH" \
    HOME="/app"

EXPOSE 8888

WORKDIR /app

USER 1001

ENTRYPOINT [ "jupyter", "notebook", "--ip", "0.0.0.0" ]
```

Build the image using:

```bash
docker build -f Dockerfile -t mlops:v0 .
```

Then you can run it with:

```bash
mkdir app
chmod -R 777 app
docker run --name jupyter -p 8888:8888 -v ./app:/app mlops:v0
```

In the logs, you will see the Jupyter URL needed to access the environment. The files you create will be stored in the `app` directory.