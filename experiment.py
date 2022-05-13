from unseen_trainer_tf import Trainer
import tensorflow as tf
from absl import app


def main(flag, k):
    trainer = Trainer(flag=flag, k=k)
    print("Start Experiment")
    trainer.train()
    print("Start Testing")
    trainer.test()


if __name__ == '__main__':
    main(flag="phone", k=1)

    # main(flag="watch", k=2)
