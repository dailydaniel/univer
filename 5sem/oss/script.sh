#!/bin/sh

find / -perm /u+s,g+s |
    while IFS= read -r line; do
        # echo "$line"
	stat --printf="%U\t%G\t%A\t%n\n" "$line"
    done
