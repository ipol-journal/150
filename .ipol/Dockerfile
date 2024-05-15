# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
FROM registry.ipol.im/ipol:v1-py3.9

# copy the code to $bin
ENV bin /workdir/bin
RUN mkdir -p $bin
WORKDIR $bin
COPY . .

#run make file and copy the binary

WORKDIR $bin/src
# Define the list of sigma values
ARG SIGMA_VALUES="10 25 30"

# Copy the source file for each sigma value, compile, and rename output
RUN for sigma in $SIGMA_VALUES; do \
        cp NeuralStruct.cpp.$sigma NeuralStruct.cpp && \
        make denoise && \
        mv denoise denoise_$sigma; \
    done && \
    cp denoise_* $bin

WORKDIR $bin/src-1
RUN make -f makefile.gcc
RUN cp imdiff $bin


# the execution will happen in the folder /workdir/exec
# it will be created by IPOL

# some QoL tweaks
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH

# $HOME is writable by the user `ipol`, but 
ENV HOME /home/ipol
# chmod 777 so that any user can use the HOME, in case the docker is run with -u 1001:1001
RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $HOME && chmod -R 777 $HOME
USER ipol