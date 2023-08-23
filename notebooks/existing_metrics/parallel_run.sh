#!/bin/bash

conda run -n q2r_metric --live-stream python eval.py &
conda run -n q2r_metric --live-stream python eval.py &
conda run -n q2r_metric --live-stream python eval.py &
wait


