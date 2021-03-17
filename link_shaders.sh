#!/bin/bash
find . -name "*.glsl" -print0 | xargs -0 -n1 -P 16 -I{} sh -c 'ln -rfs "{}" "target"'
