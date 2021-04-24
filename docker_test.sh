#!/usr/bin/env bash
# Tests if docker container works
# docker run -v $(pwd)/data:/data/ cbh2021 --data /data/CASP12_ESM1b.npz
docker run -it -v $(pwd)/data:/data/ cbh2021 --data /data/CASP12_ESM1b.npz
