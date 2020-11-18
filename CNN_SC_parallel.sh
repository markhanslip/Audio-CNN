#!/bin/bash

(echo xfce4-terminal --geometry 134x140+960+0 -x sh SC.sh; echo xfce4-terminal --geometry 134x140+0+0 -x sh CNN.sh) | parallel
