#!/bin/bash

# list of models_size
mode_list=("large" "base" "mobile")


# where or not download all size models
all_size="false"

# choose model size altenative: large/base/mobile
mode="mobile"


if [ "$all_size" = "false" ]; then
    python pyscript/download.py --models_size "$mode"
else
    for item in "${mode_list[@]}"
    do
        python pyscript/download.py --models_size "$item"
    done
fi

# 删除tensorflow的模型文件
find . -name '*.h5' -type f -delete

echo "All models downloaded..."