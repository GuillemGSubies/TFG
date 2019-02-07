import matplotlib.pyplot as plt

def plot_history(history):
    """Plots the evolution of the perplexity and the loss through the batches
    
    Parameters
    ----------
    history : keras.callbacks.History
    """

    loss_list = history.history["loss"]
    val_loss_list = history.history["val_loss"]
    perplexity_list = history.history["perplexity_raw"]
    val_perplexity_list = history.history["val_perplexity_raw"]
 
    epochs = range(1,len(history.epoch) + 1)

    ## Loss
    plt.figure(1)
    plt.plot(epochs, loss_list, 'b', label=f'Training loss ({round(loss_list[-1], 4)})')
    plt.plot(epochs, val_loss_list, 'g', label=f'Validation loss ({round(val_loss_list[-1], 4)})')
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    
    ## Perplexity
    plt.figure(2)
    plt.plot(epochs, perplexity_list, 'b', label=f'Training perplexity ({round(perplexity_list[-1], 4)})')
    plt.plot(epochs, val_perplexity_list, 'g', label=f'Validation perplexity ({round(val_perplexity_list[-1], 4)})')
    plt.title('Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend(loc='upper left')
    plt.show()