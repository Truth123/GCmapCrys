from dataset import MyDataset
from torch_geometric.loader import DataLoader as gemetric_Dataloader
from net.gat import GCmapCrys
import util
import os, argparse, torch
from tqdm import tqdm
import numpy as np
import csv
from generate_features import generate_feats

projectPath = os.path.dirname(__file__)

def run():
    parser = argparse.ArgumentParser(description='test the crystallzation network')
    parser.add_argument('conf',help='configure file')
    args = parser.parse_args()
    conf = util.load_yaml(args.conf)
    
    input_fil = os.path.join(projectPath, conf["input_file"])
    feature_dir = os.path.join(projectPath, conf["feature_dir"])

    test_dataset = MyDataset(
        input_fil, 
        feature_dir
    )
    test_no_feats_dict = {}
    for id in test_dataset.x.keys():
        feats_file = os.path.join(feature_dir, id + ".h5")
        if not os.path.exists(feats_file):
            test_no_feats_dict[id] = test_dataset.x[id]
    generate_feats(test_no_feats_dict, feature_dir)

    batch_size = conf["session"]["batch_size"]
    test_dataloader = gemetric_Dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, pin_memory = True)

    device = conf["session"]["device"]
    model = GCmapCrys(conf["GCmapCrys"])
    model.to(device)
    model.load_state_dict(torch.load(conf["checkpoint"]["load_pth"]))

    model.eval()
    te_y_id = []
    te_y_score = torch.tensor([])

    for data in tqdm(test_dataloader):
        data = data.to(device)
        out = model(data)
        te_y_id += list(data.id)
        out = out.squeeze(dim=-1)
        out = torch.sigmoid(out)
        te_y_score = torch.cat((te_y_score, out.detach().cpu()))
        
    with open(conf["output"], 'w', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(["id", "score"])
        for i, value in enumerate(te_y_score):
            writer.writerow([te_y_id[i], value])


if __name__ == "__main__":
    run()