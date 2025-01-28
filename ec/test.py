import datetime
import json
import os
import pickle
import subprocess
import sys

from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.program import Program, Invented
from dreamcoder.utilities import eprint, timing, callCompiled, get_root_dir
from dreamcoder.vs import induceGrammar_Beta

#compressor_file = os.path.join(get_root_dir(), 'compression')
compressor_file = "/Users/wangnan/Desktop/proj/dreamcoder-arc/ec/compression"
print(compressor_file)
process = subprocess.Popen(compressor_file, stdin=subprocess.PIPE, stdout=subprocess.PIPE)