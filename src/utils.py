import matplotlib.pyplot as plt
import src.config as config

def training_plot(history):

    acc = history.history.get("accuracy")
    val_acc = history.history.get("val_accuracy")
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")
    epochs = range(1, len(acc) + 1)


    plt.figure(figsize=config.FIG_SIZE)

    # --- Plot Accuracy ---

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b", label="Training-Accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation-Accuracy")
    plt.title("Accuracy-Curve", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)


    # --- Plot Loss ---

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b", label="Training-Loss")
    plt.plot(epochs, val_loss, "r", label="validation-Loss")
    plt.title("Loss-Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    
    plt.suptitle("Disaster Detection Model", fontsize=16)
    plt.savefig(config.PLOT_PATH)
    plt.close()

    print(f"✅ Training metrics plot saved at: {config.PLOT_PATH}")


def custom_training_loop_plot(history):

    acc = history.get("train_acc")
    val_acc = history.get("val_acc")
    loss = history.get("train_loss")
    val_loss = history.get("val_loss")
    epochs = range(1, len(acc) + 1)


    plt.figure(figsize=config.FIG_SIZE)

    # --- Plot Accuracy ---

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b", label="Training-Accuracy")
    plt.plot(epochs, val_acc, "r", label="Validation-Accuracy")
    plt.title("Accuracy-Curve", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)


    # --- Plot Loss ---

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "b", label="Training-Loss")
    plt.plot(epochs, val_loss, "r", label="validation-Loss")
    plt.title("Loss-Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)


    plt.suptitle("Disaster Detection Model", fontsize=16)
    plt.savefig(config.PLOT_PATH)
    plt.close()

    print(f"✅ Training metrics plot saved at: {config.PLOT_PATH}")