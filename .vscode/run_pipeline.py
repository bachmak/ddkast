"""Run all five pipeline stages in sequence. For VS Code debugging only."""

from ddkast.config import load
from ddkast.pipeline import download, evaluate, merge, predict, train

config = load()
download.run(config)
merge.run(config)
train.run(config)
predict.run(config)
evaluate.run(config)
