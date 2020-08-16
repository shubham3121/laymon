=====
Usage
=====

How to use `laymon` to monitor layers
---------------------------------
To use PyLaymon in a project::

    # Import laymon
    import laymon

    # Initialize the feature map monitoring class
    fMonitor = laymon.FeatureMapMonitoring()

    # Add your model whose feature maps are to be monitored.
    fMonitor.add_model(net)

    # To start monitoring, call the `start` method during training of the model.
    fMonitor.start()


To get the list of layers being monitored::

    fMonitor.monitor.get_registered_observers()

Instead of adding the whole model, one can add specific layers, whose feature maps are to be monitored.::

    fMonitor.add_layer(net.conv2, 'conv2')

To remove a layer from the monitoring::

    fMonitor.remove_layer(net.conv2)


Example
-------
Below is a sample example::

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from laymon import FeatureMapMonitoring

    # Creating a neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Initializing the neural network
    net = Net()

    # Initialize the feature map monitor
    fMonitor = FeatureMapMonitoring()

    # Add the neural network model to be monitored
    fmonitoring.add_model(net)

    # Add your loss and optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # Start monitoring the feature maps
        fmonitoring.start()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')


The full example notebook can be found here_.

.. _here: https://github.com/shubham3121/laymon/tree/master/examples


Custom Displays
---------------

One may feel the need to have a custom display class for visualizing the feature maps. One the applications could be displaying additional meta info or using a third partly application for visualization (e.g. Grafana) instead of the matplotlib library.

In order to accomplish this one can write his/her own custom display class and plugin it into the FeatureMapMonitoring instance.

::

    # Import the Display abstract class
    from laymon.interfaces import Display
    from laymon import FeatureMapMonitoring

    # Create your own custom class by inheriting
    # and implementing the methods of the Display class.

    class MyCustomDisplay(Display):
        def custom_method(self, params):
            // My Custom method

        def update_display(self, parameters, display_title):
            // Implement this method.
            // This method needs to be implemented,
            // as this method is invoked by the observers/layer object being monitored to
            // send the updated parameters to the display function
            // Your Custom Logic.
            // Calling your custom methods.


    # Create instance of the FeatureMapMonitoring class
    f_map_monitor = FeatureMapMonitoring()

    # Overwrite the display class with your custom display class.
    f_map_monitor.observer_factory.display_object = MyCustomDisplay

    # Now the observers/layers being monitored point to your custom display method.
