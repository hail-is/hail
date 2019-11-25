FROM ubuntu:18.04

RUN apt-get update && \
  apt-get -y install \
    openjdk-8-jdk-headless \
    wget \
  && \
  rm -rf /var/lib/apt/lists/*
# RUN wget www.scala-lang.org/files/archive/scala-2.11.8.deb && \
#     dpkg -i scala-2.11.8.deb && \
#     rm scala-2.11.8.deb
COPY shuffler-assembly-0.1.jar ./

ENTRYPOINT ["java", "-jar", "shuffler-assembly-0.1.jar"]
CMD []
