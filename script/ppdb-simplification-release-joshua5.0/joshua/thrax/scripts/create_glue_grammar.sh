#!/bin/bash
# this script just wraps a java call

if [[ -z "$THRAX" ]]
then
    THRAX="`basename $0`/.."
fi

java -cp $THRAX/bin/thrax.jar edu.jhu.thrax.util.CreateGlueGrammar $1

