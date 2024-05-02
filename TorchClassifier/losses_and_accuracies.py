import matplotlib.pyplot as plt
import numpy as np
##############################################
# cnnmodel1
##############################################
cnn_train_losses = [0.6827, 0.6537, 0.6426, 0.6204, 0.6401, 0.6020, 0.6209, 0.6011, 0.6087, 0.5947, 0.5733, 0.5738, 0.5686, 0.5738, 0.5721, 0.5622, 0.5470, 0.5451, 0.5398, 0.5269, 0.5386, 0.5228, 0.5186, 0.5135, 0.5165, 0.5251, 0.5145, 0.5116, 0.5022, 0.5361, 0.4863, 0.4726, 0.4735, 0.4624, 0.4762, 0.4643, 0.4735, 0.4552, 0.4818, 0.4613]
cnn_val_losses = [0.6551, 0.6453, 0.6045, 0.5830, 0.5909, 0.5819, 0.5796, 0.5995, 0.5677, 0.5743, 0.5525, 0.5978, 0.5698, 0.5379, 0.5524, 0.5424, 0.5244, 0.5222, 0.5192, 0.5136, 0.5081, 0.5666, 0.4986, 0.5179, 0.4967, 0.5985, 0.4934, 0.4830, 0.5045, 0.4958, 0.4862, 0.4793, 0.4793, 0.4978, 0.4792, 0.4808, 0.4802, 0.4826, 0.4753, 0.4781]
cnn_train_accuracies = [0.5560, 0.6105, 0.6205, 0.6610, 0.6310, 0.6745, 0.6465, 0.6720, 0.6695, 0.6860, 0.6875, 0.7005, 0.6940, 0.7020, 0.7115, 0.7060, 0.7330, 0.7315, 0.7330, 0.7285, 0.7200, 0.7305, 0.7460, 0.7505, 0.7360, 0.7315, 0.7465, 0.7510, 0.7575, 0.7260, 0.7530, 0.7645, 0.7745, 0.7850, 0.7740, 0.7780, 0.7760, 0.7815, 0.7590, 0.7800]
cnn_val_accuracies = [0.6250, 0.6200, 0.6940, 0.6810, 0.6920, 0.7020, 0.7060, 0.6950, 0.7110, 0.6960, 0.7240, 0.6730, 0.7100, 0.7240, 0.7080, 0.7160, 0.7380, 0.7310, 0.7470, 0.7510, 0.7510, 0.7230, 0.7580, 0.7570, 0.7630, 0.6730, 0.7530, 0.7500, 0.7540, 0.7530, 0.7500, 0.7600, 0.7580, 0.7570, 0.7650, 0.7630, 0.7580, 0.7630, 0.7620, 0.7630]
cnn_test_loss = 0.479171
cnn_test_accuracy = 76
cnn_test_accuracy_cats = 72
cnn_test_accuracy_dogs = 80

##############################################
# lenet
##############################################
# Arrays to store metrics for each epoch
lenet_train_losses = [0.6907, 0.6781, 0.6678, 0.6544, 0.6469, 0.6584, 0.6412, 0.6366, 0.6287, 0.6368, 0.6328, 0.6272, 0.6135, 0.6159, 0.6152, 0.6047, 0.6017, 0.6230, 0.5944, 0.6061, 0.6073, 0.5880, 0.5903, 0.5974, 0.5798, 0.5917, 0.5870, 0.5896, 0.5947, 0.5765, 0.5550, 0.5670, 0.5663, 0.5482, 0.5643, 0.5693, 0.5528, 0.5513, 0.5584, 0.5529]
lenet_val_losses = [0.6765, 0.6534, 0.6503, 0.6331, 0.6118, 0.6155, 0.6110, 0.6161, 0.6738, 0.5989, 0.6113, 0.6034, 0.5768, 0.5710, 0.5905, 0.5890, 0.5865, 0.5884, 0.5835, 0.5576, 0.5616, 0.5629, 0.5616, 0.5692, 0.5512, 0.6034, 0.5503, 0.5868, 0.5950, 0.5725, 0.5450, 0.5450, 0.5444, 0.5404, 0.5406, 0.5364, 0.5331, 0.5429, 0.5311, 0.5332]
lenet_train_accuracies = [0.5405, 0.5775, 0.5925, 0.6160, 0.6345, 0.6105, 0.6435, 0.6480, 0.6530, 0.6395, 0.6540, 0.6520, 0.6655, 0.6600, 0.6650, 0.6825, 0.6685, 0.6525, 0.6895, 0.6645, 0.6760, 0.6885, 0.6855, 0.6745, 0.6975, 0.6845, 0.6895, 0.6855, 0.6750, 0.7105, 0.7225, 0.7015, 0.7060, 0.7245, 0.7090, 0.6950, 0.7195, 0.7350, 0.7190, 0.7220]
lenet_val_accuracies = [0.5910, 0.6230, 0.6230, 0.6480, 0.6750, 0.6620, 0.6890, 0.6740, 0.6160, 0.6680, 0.6630, 0.6830, 0.7030, 0.7080, 0.6910, 0.6910, 0.6940, 0.7130, 0.7020, 0.7190, 0.7300, 0.7220, 0.7190, 0.7050, 0.7270, 0.6710, 0.7260, 0.6790, 0.6910, 0.7110, 0.7450, 0.7350, 0.7300, 0.7390, 0.7390, 0.7370, 0.7480, 0.7340, 0.7430, 0.7430]
lenet_test_loss = 0.533058
lenet_test_accuracy = 74
lenet_test_accuracy_cats = 71
lenet_test_accuracy_dogs = 77

##############################################
# mlp
##############################################
# Training and validation losses and accuracies
mlp_train_losses = [0.7342, 0.6860, 0.6923, 0.6818, 0.6763, 0.6708, 0.6737, 0.6747, 0.6606, 0.6617, 0.6694, 0.6635, 0.6639, 0.6623, 0.6564, 0.6618, 0.6588, 0.6656, 0.6561, 0.6580, 0.6556, 0.6565, 0.6509, 0.6549, 0.6453, 0.6437, 0.6513, 0.6501, 0.6560, 0.6481, 0.6417, 0.6360, 0.6307, 0.6313, 0.6367, 0.6292, 0.6391, 0.6320, 0.6299, 0.6293]
mlp_val_losses = [0.6957, 0.6881, 0.6729, 0.6610, 0.6710, 0.6580, 0.6577, 0.6595, 0.6601, 0.6600, 0.6590, 0.6591, 0.6507, 0.6581, 0.6555, 0.6533, 0.6515, 0.6593, 0.6445, 0.6461, 0.6531, 0.6480, 0.6431, 0.6435, 0.6521, 0.6502, 0.6473, 0.6487, 0.6522, 0.6559, 0.6527, 0.6511, 0.6360, 0.6343, 0.6322, 0.6287, 0.6295, 0.6323, 0.6311, 0.6313]
mlp_train_accuracies = [0.5395, 0.5555, 0.5570, 0.5820, 0.5725, 0.5830, 0.5735, 0.5820, 0.6100, 0.5945, 0.5875, 0.5900, 0.5865, 0.5985, 0.6025, 0.5985, 0.5955, 0.5935, 0.6055, 0.6280, 0.6045, 0.6210, 0.5870, 0.6035, 0.6205, 0.6235, 0.6160, 0.5935, 0.6200, 0.6200, 0.6345, 0.6305, 0.6295, 0.6270, 0.6265, 0.6355, 0.6300, 0.6235, 0.6315, 0.6335]
mlp_val_accuracies = [0.5340, 0.5650, 0.5890, 0.5960, 0.5590, 0.5900, 0.6100, 0.6300, 0.6000, 0.6030, 0.5940, 0.6050, 0.6120, 0.5980, 0.6130, 0.6060, 0.6120, 0.6010, 0.6140, 0.6280, 0.6130, 0.6210, 0.6300, 0.6170, 0.6200, 0.6110, 0.6140, 0.6180, 0.6300, 0.6040, 0.6220, 0.6290, 0.6280, 0.6380, 0.6450, 0.6360, 0.6380, 0.6530, 0.6440, 0.6440]
# Test variables
mlp_test_loss = 0.632267
mlp_test_accuracy = 65  # Overall percentage
mlp_test_accuracy_cats = 63  # Percentage for cats
mlp_test_accuracy_dogs = 67  # Percentage for dogs

##############################################
# alexnet
##############################################

alexnet_train_losses = [0.7002, 0.6938, 0.6922, 0.6846, 0.6703, 0.6659, 0.6678, 0.6656, 0.6686, 0.6483,0.6369, 0.6264, 0.6270, 0.6155, 0.6190, 0.6053, 0.6158, 0.6028, 0.6030, 0.5914,0.5965, 0.6008, 0.5922, 0.5738, 0.5767, 0.5788, 0.5741, 0.5820, 0.5723, 0.5783,0.5484, 0.5329, 0.5201, 0.5174, 0.5137, 0.5029, 0.5203, 0.5284, 0.5241, 0.5095]
alexnet_val_losses = [0.6913, 0.6922, 0.6897, 0.6695, 0.6617, 0.7273, 0.6218, 0.6678, 0.6330, 0.7118,0.6344, 0.5801, 0.6148, 0.6062, 0.6770, 0.6440, 0.5755, 0.5869, 0.5762, 0.5706,0.6210, 0.6216, 0.5625, 0.5628, 0.5747, 0.5852, 0.5878, 0.5441, 0.5765, 0.5648,0.5517, 0.5479, 0.5671, 0.5636, 0.5430, 0.5410, 0.5392, 0.5173, 0.5327, 0.5300]
alexnet_train_accuracies = [0.5000, 0.5055, 0.5390, 0.5670, 0.5975, 0.6050, 0.6095, 0.5910, 0.6305, 0.6395,0.6415, 0.6605, 0.6520, 0.6635, 0.6655, 0.6775, 0.6665, 0.6905, 0.6755, 0.6830,0.6825, 0.6910, 0.6885, 0.6970, 0.7045, 0.7060, 0.6990, 0.6890, 0.7070, 0.7060,0.7240, 0.7390, 0.7295, 0.7405, 0.7425, 0.7560, 0.7375, 0.7450, 0.7285, 0.7510]
alexnet_val_accuracies = [0.5280, 0.5130, 0.4990, 0.5880, 0.6380, 0.5900, 0.6440, 0.6580, 0.6710, 0.5380,0.6900, 0.7060, 0.6570, 0.6690, 0.6330, 0.6360, 0.6750, 0.7050, 0.7090, 0.7040,0.7150, 0.6050, 0.7060, 0.7190, 0.7070, 0.7110, 0.6990, 0.7120, 0.6950, 0.7030,0.7260, 0.7240, 0.7240, 0.7160, 0.7310, 0.7310, 0.7150, 0.7420, 0.7340, 0.7410]
# Test Accuracies
alexnet_test_loss = 0.517282
alexnet_test_accuracy_cats = 70  # 70% accuracy for cats
alexnet_test_accuracy_dogs = 77  # 77% accuracy for dogs
alexnet_test_accuracy = 74  # 74% overall accuracy

##############################################
# custom efficient net
##############################################
eff_train_losses = [0.4333, 0.1852, 0.1587, 0.1590, 0.1744, 0.1275, 0.1467, 0.1597, 0.1227, 0.1092]
eff_val_losses = [0.0669, 0.0637, 0.1200, 0.1979, 0.0814, 0.0269, 0.1733, 0.2050, 0.0544, 0.0407]
eff_train_accuracies = [0.8495, 0.9195, 0.9400, 0.9310, 0.9295, 0.9455, 0.9495, 0.9375, 0.9535, 0.9495]
eff_val_accuracies = [0.9810, 0.9750, 0.9710, 0.9530, 0.9710, 0.9900, 0.9560, 0.9550, 0.9870, 0.9820]

# Test Accuracies
eff_test_loss = 0.026913
eff_test_accuracy_cats = 98  # 98% accuracy for cats
eff_test_accuracy_dogs = 99  # 99% accuracy for dogs
eff_test_accuracy = 99  # 99% overall accuracy

#################################################

# Plotting the training and validation metrics
epochs = list(range(0, 40))
plt.figure(figsize=(14, 8))
mark = (0,1)
# cnnmodel1
plt.plot(epochs, cnn_train_losses, label='CNN Train Loss', marker='o', markevery=mark)
plt.plot(epochs, cnn_val_losses, label='CNN Val Loss', marker='o', markevery=mark)
plt.plot(epochs, cnn_train_accuracies, label='CNN Train Accuracy', marker='o', markevery=mark)
plt.plot(epochs, cnn_val_accuracies, label='CNN Val Accuracy', marker='o', markevery=mark)
plt.title('Training and Validation Metrics for CNN Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('cnn_model_train_val_metrics.svg', format='svg')
plt.show()

epochs = list(range(1, 41))
plt.figure(figsize=(14, 8))
# lenet
plt.plot(epochs, lenet_train_losses, label='LeNet Train Loss', marker='o', markevery=mark)
plt.plot(epochs, lenet_val_losses, label='LeNet Val Loss', marker='o', markevery=mark)
plt.plot(epochs, lenet_train_accuracies, label='LeNet Train Accuracy', marker='o', markevery=mark)
plt.plot(epochs, lenet_val_accuracies, label='LeNet Val Accuracy', marker='o', markevery=mark)
plt.title('Training and Validation Metrics for LeNet Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('lenet_model_train_val_metrics.svg', format='svg')
plt.show()

epochs = list(range(1, 41))
plt.figure(figsize=(14, 8))
# mlp
plt.plot(epochs, mlp_train_losses, label='MLP Train Loss', marker='o', markevery=mark)
plt.plot(epochs, mlp_val_losses, label='MLP Val Loss', marker='o', markevery=mark)
plt.plot(epochs, mlp_train_accuracies, label='MLP Train Accuracy', marker='o', markevery=mark)
plt.plot(epochs, mlp_val_accuracies, label='MLP Val Accuracy', marker='o', markevery=mark)
plt.title('Training and Validation Metrics for MLP Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('mlp_model_train_val_metrics.svg', format='svg')
plt.show()

epochs = list(range(1, 41))
plt.figure(figsize=(14, 8))
# alexnet
plt.plot(epochs, alexnet_train_losses, label='AlexNet Train Loss', marker='o', markevery=mark)
plt.plot(epochs, alexnet_val_losses, label='AlexNet Val Loss', marker='o', markevery=mark)
plt.plot(epochs, alexnet_train_accuracies, label='AlexNet Train Accuracy', marker='o', markevery=mark)
plt.plot(epochs, alexnet_val_accuracies, label='AlexNet Val Accuracy', marker='o', markevery=mark)
plt.title('Training and Validation Metrics for AlexNet Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('alex_net_train_val_metrics.svg', format='svg')
plt.show()

###############################################################################

import matplotlib.pyplot as plt
import numpy as np

models = ['CNN', 'LeNet', 'MLP', 'AlexNet', 'EfficientNet']
test_accuracies = [cnn_test_accuracy, lenet_test_accuracy, mlp_test_accuracy, alexnet_test_accuracy, eff_test_accuracy]
test_accuracies_cats = [cnn_test_accuracy_cats, lenet_test_accuracy_cats, mlp_test_accuracy_cats, alexnet_test_accuracy_cats, eff_test_accuracy_cats]
test_accuracies_dogs = [cnn_test_accuracy_dogs, lenet_test_accuracy_dogs, mlp_test_accuracy_dogs, alexnet_test_accuracy_dogs, eff_test_accuracy_dogs]
colors = ['blue', 'green', 'red', 'purple', 'orange']

bar_width = 0.2  # width of the bars
index = np.arange(len(models))  # the label locations

plt.figure(figsize=(14, 8))
bar1 = plt.bar(index - bar_width, test_accuracies, bar_width, label='Overall Accuracy', color='grey')
bar2 = plt.bar(index, test_accuracies_cats, bar_width, label='Cats Accuracy', color='cyan')
bar3 = plt.bar(index + bar_width, test_accuracies_dogs, bar_width, label='Dogs Accuracy', color='magenta')

plt.xlabel('Model')
plt.ylabel('Test Accuracy (%)')
plt.title('Comparison of Test Accuracies Across Models')
plt.xticks(index, models)
plt.ylim([60, 100])  # Adjust this range based on the data
plt.grid(True)
plt.legend()

plt.savefig('test_accuracies_comparison.svg', format='svg')
plt.show()

test_losses = [cnn_test_loss, lenet_test_loss, mlp_test_loss, alexnet_test_loss, eff_test_loss]  # Assuming you add the EfficientNet loss here

plt.figure(figsize=(14, 8))
plt.plot(models, test_losses, color='black', marker='o', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Model')
plt.ylabel('Test Loss')
plt.title('Comparison of Test Losses Across Models')
plt.grid(True)

plt.savefig('test_losses_comparison.svg', format='svg')
plt.show()

###################################################################################

# Plotting the training and validation metrics for the custom EfficientNet
epochs_eff = list(range(1, 11))
plt.figure(figsize=(10, 6))

plt.plot(epochs_eff, eff_train_losses, label='EfficientNet Train Loss', marker='o', markevery=1)
plt.plot(epochs_eff, eff_val_losses, label='EfficientNet Val Loss', marker='o', markevery=1)
plt.plot(epochs_eff, eff_train_accuracies, label='EfficientNet Train Accuracy', marker='o', markevery=1)
plt.plot(epochs_eff, eff_val_accuracies, label='EfficientNet Val Accuracy', marker='o', markevery=1)

plt.title('Training and Validation Metrics for CustomEfficientNet')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('efficient_net_train_val_metrics.svg', format='svg')
plt.show()