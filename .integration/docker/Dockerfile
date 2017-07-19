# build
FROM index.docker.io/library/debian:latest
MAINTAINER qinka <me@qinka.pro>
RUN apt update && apt install -y libgmp10 curl
ADD bin /usr/bin
COPY entrypoint.sh entrypoint
ENTRYPOINT ["/entrypoint"]
EXPORT 3000
