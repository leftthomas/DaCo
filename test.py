import torch

vector_dict = torch.load('results/npid_alderley_128_0.07_64_200_4096_0.5_200_vectors.pth')

image_names, vectors = [], []
for image_name, vector in vector_dict.items():
    image_names.append(image_name)
    vectors.append(vector)
vectors = torch.stack(vectors, dim=0)
sim_matrix = torch.mm(vectors, vectors.t())
sim_matrix.fill_diagonal_(-1)
idx = sim_matrix.topk(k=5, dim=-1, largest=True)[1]
top_k_dict = {}
for index in range(len(image_names)):
    top_k_dict[image_names[index]] = [image_names[i] for i in idx[index].cpu().tolist()]
print(top_k_dict)
