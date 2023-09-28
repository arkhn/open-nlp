#! /bin/bash
grep -Eo '\[\*.*?\*.*?\*.*?\* ?]' "$1" | sed s/.*://g | sort | uniq
