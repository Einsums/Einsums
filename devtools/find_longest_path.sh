#!/bin/bash

# I'm including this to help devs determine issues relating to path lengths.

set line_limit=260

if [[ "$1" != "" ]]; then
  line_limit=$1
fi

if [[ "$line_limit" == "" ]] then
  echo "Something went wrong setting the line limit."
  exit
fi

set max=

for i in $(find ${PWD} -printf "${PWD}/%P\n"); do
  if [[ ${#i} -ge ${line_limit} ]]; then
    echo "The path $i is too long at ${#i} characters."
  fi
  
  if [[ ${#i} -gt ${#max} ]]; then
    max=$i
  fi
done

echo "The longest path was $max at ${#max} characters"
