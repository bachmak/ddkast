"""Run all pipeline stages in sequence. For debugging only."""

from ddkast.config import load
from ddkast.pipeline import download, evaluate, merge, predict, train, visualise

config = load()
download.run(config)
merge.run(config)
train.run(config)
predict.run(config)
evaluate.run(config)
visualise.run(config)
