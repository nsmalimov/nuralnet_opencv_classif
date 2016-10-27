from tensorflow.models.image.cifar10 import cifar10_input

data_dir = "/cifar10_tf/cifar10_data/cifar-10-batches-bin/"



images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=128)

images1, labels1 = cifar10_input.read_train_images()

print (images)

print (images == images)

print (labels)