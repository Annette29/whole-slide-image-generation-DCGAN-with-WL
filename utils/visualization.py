import matplotlib.pyplot as plt

def plot_losses(c_losses, g_losses, num_epochs, start_epoch, save_path):
    # Create x-axis values corresponding to each epoch
    x_values = [epoch + start_epoch for epoch in range(num_epochs)]

    # Create a 1x2 grid for 2 subplots
    plt.figure(figsize=(12, 6))

    # Plot the Generator Losses
    plt.subplot(1, 2, 1)
    plt.plot(x_values, g_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator Losses')

    # Plot the Critic Losses
    plt.subplot(1, 2, 2)
    plt.plot(x_values, c_losses, label='Critic Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Critic Losses')

    # Save the Critic and Generator Losses
    plt.savefig(save_path)

    # Show the plots
    plt.show()
