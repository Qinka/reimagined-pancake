# build
FROM index.docker.io/library/ubuntu:zesty
MAINTAINER qinka <me@qinka.pro>
RUN apt update && apt install -y libgmp10 curl proto-compiler
ADD bin /usr/bin
ADD libtensorflow /usr/local
COPY entrypoint.sh entrypoint
ENTRYPOINT ["/entrypoint"]
EXPOSE 3000
