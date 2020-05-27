import matplotlib.pyplot as plt
import numpy as np

def display_predictions(training_images, preds, training_truths=[], submission_outputs=[], samples=5):
    training_images = np.array(training_images)
    preds = np.array(preds)
    training_truths = np.array(training_truths)
    submission_outputs = np.array(submission_outputs)
    indices = np.random.choice(len(training_images), samples, replace=False)
    
    dim = preds[0].shape[0]
    columns = 3
    if len(training_truths) == 0 and len(submission_outputs) == 0:
        columns = 2
    for i in range(samples):
        ax1 = plt.subplot2grid((samples, columns), (i, 0))
        ax1.imshow(training_images[indices][i])
        ax1.axis('off')
        ax1.set_title('Original Image')
        
        ax2 = plt.subplot2grid((samples, columns), (i, 1))
        ax2.imshow(preds[indices][i].reshape(dim, dim), cmap= 'Greys_r')
        ax2.axis('off')
        ax2.set_title('Predicted Mask')
        
        if len(training_truths) != 0:
            ax3 = plt.subplot2grid((samples, columns), (i, 2))
            ax3.imshow(training_truths[indices][i].reshape(dim, dim), cmap= 'Greys_r')
            ax3.axis('off')
            ax3.set_title('Original Mask')
            
        if len(submission_outputs) != 0:
            ax3 = plt.subplot2grid((samples, columns), (i, 2))
            ax3.imshow(submission_outputs[indices][i].reshape(dim, dim), cmap= 'Greys_r')
            ax3.axis('off')
            ax3.set_title('Submission Output')