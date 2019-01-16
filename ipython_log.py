# IPython log file

runfile('C:/Users/karan.verma/.spyder-py3/deep-learning/digits_mnist_trial_with_convnets.py', wdir='C:/Users/karan.verma/.spyder-py3/deep-learning')
print('Random samples from the test data: ')
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for i in range(1, 7):
        c = np.random.choice(len(test_x_reshaped))
        plt.subplot(2,3,i)
        plt.imshow(test_x[c])
        plt.title('Original {} & Predicted {}'.format(label_list[np.argmax(test_y[c])], label_list[np.argmax(prediction[c])]))        
        #plt.tight_layout()
os.execvpe('%logstart -o')
