from unseen_trainer_tf import Trainer
import sys


def main(flag, k, data_stored_path):
    trainer = Trainer(flag=flag, k=k, data_stored_path=data_stored_path)
    print("Start Experiment")
    trainer.train()
    print("Start Testing")
    trainer.test()


if __name__ == '__main__':
    # main(flag="phone", k=1)
    k = int(sys.argv[1])
    data_stored_path = sys.argv[2]
    main(flag="watch", k=k, data_stored_path=data_stored_path)
