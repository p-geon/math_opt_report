FROM python:3.9.6
LABEL maintainer='p-geon'

RUN pip install -q --upgrade pip && \
    pip install -q \
        numpy==1.22.4

RUN pip install -q \
        matplotlib==3.1.3

WORKDIR /work
USER root
CMD ["/bin/bash"]