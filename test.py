from model import CAGNN
import numpy as np
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CAGNN(feature_size = 200,layer=3,nclass=4,top_k=0.6).to(device)
data = np.load('datapath', allow_pickle=True).item()
def test(model, modelpath, data):
    # Load the model state dictionary from the specified file path
    model.load_state_dict(torch.load(modelpath))
    # Set the model to evaluation mode to disable gradient computation
    model.eval()
    # Initialize a list to store predicted labels
    pred = []
    # Define the number of top important nodes to select
    top_k = 10
    # Initialize a list to store the indices of important nodes
    important_nodes_list = []
    
    # Iterate over 20 test samples
    for i in range(20):
        # Convert and move the i-th sample data to a PyTorch tensor on the specified device (e.g., GPU)
        X_i, A_i, Z_i, A_P_i = [torch.tensor(data[key][i]).float().to(device) for key in ['pearsons', 'edge_arrs', 'populations', 'adj_p']]
        # Perform forward propagation through the model
        outputs, _, _, final_A, S_list = model(X_i, A_i, Z_i, A_P_i)
        # Apply softmax to the model outputs along dimension 1 to obtain probability distributions
        softmax_outputs = F.softmax(outputs, dim=1)
    
        # Retrieve the index of the class with the highest probability as the predicted label
        _, predicted = torch.max(softmax_outputs, 1)
    
        # Append the predicted label (as a scalar value) to the prediction list
        pred.append(predicted[0].item())
        with torch.no_grad():
            # Extract the soft assignment matrices and final adjacency matrix for the first sample in the batch
            S_L = [S[0].cpu().detach().numpy() for S in S_list]  # Shape per layer: (b, N, n) → select the first sample
            A_L = final_A[0].cpu().detach().numpy()  # Shape of final_A: (b, n, n) → select the first sample
    
            # 1. Compute the importance scores of nodes in the final layer
            I_L = np.sum(A_L, axis=1)  # Shape: (n_L,)
    
            # 2. Combine the pooling matrices (S_1 × S_2 × S_3)
            S_combined = S_L[0]
            for l in range(1, len(S_L)):
                S_combined = np.dot(S_combined, S_L[l])  # Shape: (N, n_L)
    
            # 3. Propagate the importance scores back to the original nodes
            I_0 = np.dot(S_combined, I_L)  # Shape: (N,)
    
            # 4. Identify the top-k most important nodes based on their importance scores
            top_k_indices = np.argsort(I_0)[::-1][:top_k]
            top_k_scores = I_0[top_k_indices]
            important_nodes_list.append(top_k_indices)
    # Return the list of predicted labels and the list of important node indices
    return pred, important_nodes_list
pred,important_nodes_list = test(model,'model.pth',data)
