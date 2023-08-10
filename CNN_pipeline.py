import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from CDR_preprocessing import *
from aggregated_features import *
from input_generation import *
from CNN_architecture import *

def generate_dataloaders(interactions_file_path, label_file_path = None, bs = 8, max_x = 260, max_y = 277, test_size = 0.25):
    """
    This function takes as an argument labelled or unlabelled data and generates pytorch dataloader to feed the neural network and either train it, test it or make predictions.
    Parameters:
    -----------
    interactions_file_path : string
        The relative or absolute path towards the interactions dataframe
    label_file_path : string
        The relative or absolute path towards the label dataframe
        If the latter exists, it should contains the collowing columns:
            - hashed_imsi: string
            - label: int 0 or 1
        If the goal is to make a prediction on unlabelled data, this argument should be kept as None
    bs : int
        The batch size (default = 8)
    max_x : int
        The maximum value of the x (converted longitude) dimension in the array (default = 260)
    max_y : int
        The maximum value of the y (converted latitude) dimension in the array (default = 277)
    test_size : float
        A proportion between 0 and 1 that indicates the relative size of the test set in case of labelled data (default = 0.25)
        
    Returns:
    --------
    If label_file_path is not None:
        train_dataloader: torch.utils.data.dataloader.DataLoader
            A pytorch dataloader with labelled data for training
        test_dataloader: torch.utils.data.dataloader.DataLoader
            A pytorch dataloader with labelled data for testing
    
    If label_file_path is None:
        dataloader: torch.utils.data.dataloader.DataLoader
        A pythorch dataloader with unlabelled data for predictions
    """
    interactions = pd.read_csv(f'{interactions_file_path}')
    if label_file_path is not None:
        label_dictionnary = pd.read_csv(f'{label_file_path}', index_col=False).set_index('hashed_imsi').to_dict()['label']
    else:
        label_dictionnary = None

    input_dataframe = input_pipeline(interactions, label_dictionnary, max_x = max_x, max_y = max_y)

    if label_file_path is not None:
        train_df, test_df = train_test_split(input_dataframe, test_size=test_size, random_state=15) #random state in fixed to have the same distribution every time
        x_train = np.array(train_df['array'].to_list())
        x_test = np.array(test_df['array'].to_list())
        y_train = train_df['label'].to_numpy()
        y_test = test_df['label'].to_numpy()
        # assuming `x_train` and `y_train` are your training data and labels as NumPy arrays
        train_out_of_batch = y_train.shape[0] % bs
        test_out_of_batch = y_test.shape[0] % bs

        x_train_tensor = torch.from_numpy(x_train)[:-train_out_of_batch,:,:]
        y_train_tensor = torch.from_numpy(y_train)[:-train_out_of_batch] #modify the train and test size in order to fit with the batch size
        x_test_tensor = torch.from_numpy(x_test) 
        x_test_tensor = x_test_tensor[:-test_out_of_batch,:,:]
        y_test_tensor = torch.from_numpy(y_test)
        y_test_tensor = y_test_tensor[:-test_out_of_batch]

        # create a TensorDataset from the training data and labels
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

        # create a DataLoader from the TensorDataset
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        # create a DataLoader from the TensorDataset
        test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        return train_dataloader, test_dataloader
    
    else:
        x_predict = np.array(input_dataframe['array'].to_list())
        x_out_of_batch = x_predict.shape[0] % bs
        x_tensor = torch.from_numpy(x_predict)[:-x_out_of_batch,:,:]
        data = TensorDataset(x_tensor)
        dataloader = DataLoader(data, batch_size=bs, shuffle=False)
        return dataloader

def train(model,data_loader,loss_fn,optimizer,n_epochs=1):
    """
    This function trains the model that is fed on GPU preferentially.
    It has to be noted that the classification model has to be from the CNN_architecture.classifier class.

    Parameters:
    -----------
    model: CNN_architecture.classifier
        The classification model that is trained
    data_loader: torch.utils.data.dataloader.DataLoader
        The dataloader that contains the labelled training set
    loss_fn: torch.nn loss function
        The function that is used to compute the loss between the output of the model and the target
        ex: nn.CrossEntropyLoss()
    optimizer: torch.optim
        The optimizer that is used to process the gradient descent
        ex: torch.optim.Adam(conv_class.parameters(),lr=1e-3)
    n_epochs : int
        The number of epochs in the training
        
    Returns:
    --------
    model: classifier
        The trained classification model
    loss_train: list
        The list of the average loss of each epoch
    acc_train: list
        The list of average accuracy of each epoch
    """
    device = 'cuda' if torch.cuda else 'cpu'
    model =  model.to(device)
    model.train(True)
    loss_train = np.zeros(n_epochs)
    acc_train = np.zeros(n_epochs)
    for epoch_num in range(n_epochs):
        running_corrects = 0.0
        running_loss = 0.0
        size = 0

        for data in data_loader:
            x_train, y_train = data
            x_train = x_train.to(device)
            x_train = x_train.type(torch.float32)
            x1_train = x_train[:,:-1,:]
            x2_train = x_train[:,-1,:10]
            y_train = y_train.type(torch.LongTensor)
            y_train =  y_train.to(device)
            
            bs = x_train.size(0)

            outputs = model(x1_train, x2_train)
            loss = loss_fn(outputs,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _,preds = torch.max(outputs,1)
            running_corrects += torch.sum(preds == y_train)
            running_loss += loss.data
            size += bs

        epoch_loss = running_loss.item() / size
        epoch_acc = running_corrects.item() / size
        loss_train[epoch_num] = epoch_loss
        acc_train[epoch_num] = epoch_acc
        print('Train - Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return model, loss_train, acc_train

def test(model,data_loader, loss_fn):
    """
    This function tests a model on GPU preferentially and returns predicted and true labels in order to compute statistics

    Parameters:
    -----------
    model: CNN_architecture.classifier
        The classification model that is tested (preferentially trained)
    data_loader: torch.utils.data.dataloader.DataLoader
        The dataloader that contains the labelled test set
    loss_fn: torch.nn loss function
        The function that is used to compute the loss between the output of the model and the target
        ex: nn.CrossEntropyLoss()
        
    Returns:
    --------
    running_loss: float
        The cumulated loss over the test set
    predicted_class: np.array
        An array containing the predicted labels for each imsi in the test set
    y_test_final_np: np.array
        An array containing the true labels for each imsi in the test set
    output_cumulated: np.array
        An array containing the outputs of the neural network for each imsi in the same order than predicted_class and y_test_final_np
    """
    device = 'cuda' if torch.cuda else 'cpu'
    bs = data_loader.batch_size
    y_test_final = torch.empty(bs*len(data_loader))
    output_cumulated = torch.empty((bs*len(data_loader),2))
    predicted_class = torch.empty((bs*len(data_loader)))
    model.train(False)
    counter = 0
    running_corrects = 0.0
    running_loss = 0.0
    for data in data_loader:
        x_test_bs, y_test_bs = data
        x_test_bs = x_test_bs.to(device)
        x_test_bs = x_test_bs.type(torch.float32)
        y_test_bs = y_test_bs.type(torch.LongTensor)
        x1_test = x_test_bs[:,:-1,:]
        x2_test = x_test_bs[:,-1,:10]
        y_test_bs =  y_test_bs.to(device)
        outputs = model(x1_test, x2_test)
        loss = loss_fn(outputs,y_test_bs)
        _,preds = torch.max(outputs,1)
        running_corrects += torch.sum(preds == y_test_bs)
        running_loss += loss.data
        output_cumulated[counter:counter+bs] = outputs.detach()
        y_test_final[counter:counter+bs]= y_test_bs
        predicted_class[counter:counter+bs]= preds
        counter += bs
    print('Test - Loss: {:.4f} Acc: {:.4f}'.format(running_loss / counter, running_corrects.item() / counter))
    y_test_final_np, output_cumulated = y_test_final.to('cpu').numpy(), output_cumulated.to('cpu').numpy()
    return running_loss, predicted_class, y_test_final_np, output_cumulated

def predict(model, dataloader):
    """
    This function issues a prediction on unlabelled data using a classification model of CNN_architecure.classifier class.

    Parameters:
    -----------
    model: CNN_architecture.classifier
        The classification model through which the data is processed (preferentially trained)
    data_loader: torch.utils.data.dataloader.DataLoader
        The dataloader that contains the unlabelled set
        
    Returns:
    --------
    prediction: np.array
        An array containing the predicted labels for each imsi in the test set
    output_cumulated: np.array
        An array containing the outputs of the neural network for each imsi in the same order than predicted_class and y_test_final_np
    """

    bs = dataloader.batch_size
    output_cumulated = torch.empty((bs*len(dataloader),2))
    prediction = torch.empty((bs*len(dataloader)))
    
    model.train(False)
    counter = 0
    size = 0
    for data in dataloader:

        x_predict = data
        x_predict = x_predict[0].to(device)
        x_predict = x_predict.type(torch.float32)
        x1_predict = x_predict[:,:-1,:]
        x2_predict = x_predict[:,-1,:10]

        outputs = model(x1_predict, x2_predict)
        _,preds = torch.max(outputs,1)

        size += bs
        prediction[counter:counter+bs]= preds
        output_cumulated[counter:counter+bs] = outputs.detach()
        counter += bs
    return prediction, output_cumulated