import sys
import torch
from model import Model, Encoder, Decoder
import os

sys.path.append(os.path.abspath(".."))
if __name__ == "__main__":
    # print(sys.argv)
    
    # exp1 = sys.argv[1]
    # exp2 = sys.argv[2]
    
    # print(f"Comparing run {exp1} to {exp2}")
    
    model1 = torch.load("outputs/2022-01-05/14-34-15/trained_model.pt")
    model2 = torch.load("outputs/2022-01-05/14-34-15/trained_model.pt")
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), \
            "encountered a difference in parameters, your script is not fully reproduceable"
