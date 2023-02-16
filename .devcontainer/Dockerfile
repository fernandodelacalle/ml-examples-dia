FROM mcr.microsoft.com/vscode/devcontainers/python:3.8

RUN curl -fsSL https://deb.nodesource.com/setup_current.x | sudo -E bash -
RUN apt-get install -y nodejs

COPY requirements.txt /tmp/
RUN set -ex; \
    pip3 --no-cache-dir install --requirement /tmp/requirements.txt

ENTRYPOINT ["/workspace/.devcontainer/docker-entrypoint.sh"]
