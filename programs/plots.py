import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_history_pro(history, args):
    # Set Seaborn style
    sns.set(style="whitegrid")

    train_data = history[history['Data_Type'] == 'Train']
    test_data = history[history['Data_Type'] == 'Test']

    # Create subplots with a single row and four columns
    fig, axs = plt.subplots(2, 3, figsize=(22, 10))  # Adjust figsize for smaller plots

    # Plot 1: Basic learning plot
    sns.lineplot(data=train_data, x='Epoch', y='loss', label='Train Loss', ax=axs[0,0])
    sns.lineplot(data=train_data, x='Epoch', y='accuracy', label='Train Accuracy', ax=axs[0,0])
    sns.lineplot(data=test_data, x='Epoch', y='loss', label='Test Loss', ax=axs[0,0])
    sns.lineplot(data=test_data, x='Epoch', y='accuracy', label='Test Accuracy', ax=axs[0,0])

    axs[0,0].set_title('Train/Test history')
    axs[0,0].set_xlabel('Epoch')
    axs[0,0].set_ylabel('Loss / Accuracy')

    # Plot 2: Train Loss, Train Loss of Correct Samples, Train Loss of Incorrect Samples
    sns.lineplot(data=train_data, x='Epoch', y='loss', label='Train Loss', ax=axs[0,1])
    sns.lineplot(data=train_data, x='Epoch', y='correct_loss', label='Train Loss (Correct)', ax=axs[0,1])
    sns.lineplot(data=train_data, x='Epoch', y='incorrect_loss', label='Train Loss (Incorrect)', ax=axs[0,1])
    axs[0,1].set_title('Train Loss')
    axs[0,1].set_xlabel('Epoch')
    axs[0,1].set_ylabel('Loss / Entropy')

    # Plot 3: Test Loss, Test Loss of Correct Samples, Test Loss of Incorrect Samples
    sns.lineplot(data=test_data, x='Epoch', y='loss', label='Test Loss', ax=axs[0,2])
    sns.lineplot(data=test_data, x='Epoch', y='correct_loss', label='Test Loss (Correct)', ax=axs[0,2])
    sns.lineplot(data=test_data, x='Epoch', y='incorrect_loss', label='Test Loss (Incorrect)', ax=axs[0,2])
    axs[0,2].set_title('Test Loss')
    axs[0,2].set_xlabel('Epoch')
    axs[0,2].set_ylabel('Loss / Entropy')

    # Plot 4: Train Entropies
    sns.lineplot(data=train_data, x='Epoch', y='entropy_correct', label='Train Entropy (Correct)', ax=axs[1,0])
    sns.lineplot(data=train_data, x='Epoch', y='entropy_incorrect', label='Train Entropy (Incorrect)', ax=axs[1,0])
    axs[1,0].set_title('Train Entropy')
    axs[1,0].set_xlabel('Epoch')
    axs[1,0].set_ylabel('Entropy')

    # Plot 5: Test Entropies
    sns.lineplot(data=test_data, x='Epoch', y='entropy_correct', label='Test Entropy (Correct)', ax=axs[1,1])
    sns.lineplot(data=test_data, x='Epoch', y='entropy_incorrect', label='Test Entropy (Incorrect)', ax=axs[1,1])
    axs[1,1].set_title('Test Entropy')
    axs[1,1].set_xlabel('Epoch')
    axs[1,1].set_ylabel('Entropy')

    # Plot 6.1: Weight Norm
    sns.lineplot(data=train_data, x='Epoch', y='weight_norm', label='Weight Norm', ax=axs[1,2])
    axs[1,2].set_title('Weight Norm')
    axs[1,2].set_xlabel('Epoch')
    axs[1,2].set_ylabel('Weight Norm')

    # Create a secondary y-axis for axs[1,2]
    ax2 = axs[1,2].twinx()
    # Plot 6.2: Learning rate
    sns.lineplot(data=train_data, x='Epoch', y='lr', label='Learning Rate', ax=ax2, color='orange')
    ax2.set_ylabel('Learning Rate')

    # Flatten the axs array
    axs_flat = axs.flatten()

    # Iterate over all subplots
    for ax in axs_flat:
        ax.grid(True)
        ax.legend(loc='best')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{args.model_folder}/{args.model_name}_history.png', dpi=300, bbox_inches='tight')

    # Show the plots
    plt.close()
