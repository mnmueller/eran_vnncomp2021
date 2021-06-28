#!/bin/bash

set -e

VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

if [ $(id -u) = 0 ]; then
   echo "The script must not be run as root."
   exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

sudo -E bash "$SCRIPT_DIR/install_tool_sudo.sh"
bash "$SCRIPT_DIR/install_tool_user.sh"