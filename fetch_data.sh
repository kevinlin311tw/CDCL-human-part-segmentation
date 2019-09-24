#!/bin/bash

# Download pretrained networks
wget https://students.washington.edu/kvlin/data/cdcl_pascal_model.zip --directory-prefix=weights
unzip weights/cdcl_pascal_model.zip -d weights && rm weights/cdcl_pascal_model.zip

