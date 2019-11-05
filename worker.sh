#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPT=${DIR}'/../code/worker.py'

nohup /usr/bin/python $script &
if [ "$1" == "start" ]; then
    nohup python $SCRIPT $2 $3 > worker.log &
    echo 'Started Cerebro worker on host: $2 port: $3'
else
    PID=$(ps aux | grep -v grep | grep $SCRIPT | awk '{print $2}')
    kill -9 $PID
    echo 'Stopped Cerebro worker'
fi
