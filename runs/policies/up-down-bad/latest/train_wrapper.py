"""Run lerobot-train 0.6.1 on Windows without the 'last' checkpoint symlink (needs a privilege this account lacks)."""
import lerobot.scripts.lerobot_train as lerobot_train

if __name__ == "__main__":
    lerobot_train.update_last_checkpoint = lambda checkpoint_dir: checkpoint_dir
    lerobot_train.main()
