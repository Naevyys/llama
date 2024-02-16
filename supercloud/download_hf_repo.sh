#!/bin/bash

# Parse the command line arguments
while getopts "i:e:" option; do
    case "${option}" in
        i)
            include="${OPTARG}"
            ;;
        e)
            exclude="${OPTARG}"
            ;;
        *)
            ;;
    esac
done

# Get the repo argument
repo="${@: -1}"

# Invoke the 'huggingface-cli download' command
huggingface-cli download "${repo}" --include "${include}" --exclude "${exclude}"