FROM python:3.9.6
LABEL maintainer='p-geon'

COPY requirements.txt ./
RUN pip install -q --upgrade pip
RUN pip install -r requirements.txt -q


WORKDIR /work
USER root
CMD ["/bin/bash"]