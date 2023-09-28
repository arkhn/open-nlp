#!/bin/bash
egrep -o '\[\*.*?\*.*?\*.*?\* ?]' $1 | sed s/.*://g | sort | uniq > all_labels
paste -d ',' all_labels all_labels
rm all_labels
