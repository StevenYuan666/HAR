from unseen_trainer_tf import Trainer
import tensorflow as tf
from absl import app


def main():
    trainer = Trainer(flag="phone")
    print("Start Experiment")
    trainer.train()


if __name__ == '__main__':
    main()
