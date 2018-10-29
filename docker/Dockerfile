FROM ubuntu:18.04
MAINTAINER lmok

ENV HOME /opt
WORKDIR /opt

RUN apt-get update && \
  apt-get install -y python3-pip python3-dev && \
  cd /usr/local/bin && \
  ln -s /usr/bin/python3 python && \
  pip3 install --upgrade pip setuptools 

RUN mkdir /opt/SmartStretching && \
	mkdir /opt/SmartStretching.egg-info

COPY SmartStretching/* /opt/SmartStretching/
COPY SmartStretching.egg-info/* /opt/SmartStretching.egg-info/
COPY SmartStretching.ini /opt/
COPY setup.py /opt/
COPY README.rst /opt/

RUN pip install -e ".[testing]"

CMD pserve SmartStretching.ini