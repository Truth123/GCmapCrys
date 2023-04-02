from dataset import MyDataset
from torch.utils.data import DataLoader as default_Dataloader
from torch_geometric.loader import DataLoader as gemetric_Dataloader
from net.deepCrystal import DeepCrystal
from net.gat import GCmapCrys
import util
import os, argparse, torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from generate_features import generate_feats

projectPath = os.path.dirname(__file__)
MODEL = {
    "GCmapCrys": GCmapCrys,
    "DeepCrystal": DeepCrystal
}

if __name__ == "__main__":

    ## 1. load train/test config file
    parser = argparse.ArgumentParser(description='train the crystallzation network')
    parser.add_argument('conf',help='configure file')
    args = parser.parse_args()
    conf = util.load_yaml(args.conf)
    print(conf)
    

    ## 2. set params
    dataset_dir = os.path.join(projectPath, conf["dataset_dir"])
    feature_dir = os.path.join(projectPath, conf["feature_dir"])

    model_name = conf["model"]
    assert model_name in MODEL.keys()
    use_ccmap = True if model_name == 'GCmapCrys' else False

    batch_size = conf["session"]["batch_size"]
    device = conf["session"]["device"]

    ## 3. load datasets
    tr_dataset = MyDataset(
        os.path.join(dataset_dir, 'train'), 
        feature_dir, 
        use_ccmap=use_ccmap
    )
    val_dataset = MyDataset(
        os.path.join(dataset_dir, 'val'), 
        feature_dir, 
        use_ccmap=use_ccmap
    )

    ## check input features of tr_dataset
    tr_no_feats_dict = {}
    for id in tr_dataset.x.keys():
        feats_file = os.path.join(feature_dir, id + ".h5")
        if not os.path.exists(feats_file):
            tr_no_feats_dict[id] = tr_dataset.x[id]
    generate_feats(tr_no_feats_dict, feature_dir)

    
    ## check input features of val_dataset
    val_no_feats_dict = {}
    for id in val_dataset.x.keys():
        feats_file = os.path.join(feature_dir, id + ".h5")
        if not os.path.exists(feats_file):
            val_no_feats_dict[id] = val_dataset.x[id]
    generate_feats(val_no_feats_dict, feature_dir)

    
    Dataloader = gemetric_Dataloader if use_ccmap else default_Dataloader
    tr_dataloader = Dataloader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    val_dataloader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 4, pin_memory = True)

    
    model = MODEL[model_name](conf[model_name])
    model.to(device)

    ## save model dir
    save_dir = os.path.join(projectPath, conf['checkpoint']['save_dir'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_val_mcc = 0
    saved_epoch_num = 0

    ## load pre model
    if conf['checkpoint']['load']:
        print("load pth")
        model.load_state_dict(torch.load(conf['checkpoint']['load_pth']))

    ## log_file and tensorboard
    tensorboard_writer = SummaryWriter(save_dir)
    log_file = os.path.join(save_dir,"log.txt")

    ## criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=conf['session']['lr'], weight_decay=conf['session']['weight_decay'])
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 50],gamma=0.5)
    max_epoch = conf['session']['max_epochs']
    global_step = 0
    loss = 0
    for epoch in range(max_epoch):
        model.train()
        for step, data in enumerate(tr_dataloader):
            if model_name == "GCmapCrys":
                data = data.to(device)
                id = data.id
                y = data.y
                out = model(data)
            else:
                x_feature, x_emb, y, id = data
                x_feature = x_feature.to(device)
                x_emb = x_emb.to(device)
                y = y.to(device)
                out = model(x_emb, x_feature)
            out = out.squeeze(dim=-1)
            y = y.float()
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ## log loss
            if step % 100 == 0 or step == len(tr_dataloader) - 1:
                print("loss=%.7f    %d/%d"%(loss.item(), step, len(tr_dataloader)))
                tensorboard_writer.add_scalar('loss', loss.item(), global_step)
            global_step += 1
        
        scheduler.step()


        ## evaluate dataset
        print("evaluate train dataset ...")
        mcc_threshold, tr_TN, tr_FN, tr_FP, tr_TP, tr_Sen, tr_Spe, tr_Acc, tr_mcc, tr_AUC = util.evaluate(model, tr_dataloader, device)
        print('mcc_threshold=%.3f  tr_Sen=%.3f  tr_Spe=%.3f  tr_Acc=%.3f  tr_MCC=%.3f  tr_AUC=%.3f'%(
                mcc_threshold,tr_Sen, tr_Spe, tr_Acc, tr_mcc, tr_AUC))
        print("evaluate val dataset ...")
        val_TN, val_FN, val_FP, val_TP, val_Sen, val_Spe, val_Acc, val_mcc, val_AUC = util.evaluate(model, val_dataloader, device, False, mcc_threshold)
        print('val_Sen=%.3f  val_Spe=%.3f  val_Acc=%.3f  val_MCC=%.3f  val_AUC=%.3f'%(val_Sen, val_Spe, val_Acc, val_mcc, val_AUC))
        
        ## save best model
        if val_mcc > best_val_mcc:
            print("save model ...")
            save_file = os.path.join(save_dir,'model_%d.pth'%(epoch+1))
            torch.save(model.state_dict(), save_file)
            if saved_epoch_num > 0:
                rm_file = os.path.join(save_dir,'model_%d.pth'%(saved_epoch_num))
                if os.path.exists(rm_file):
                    os.system("rm %s"%(rm_file))
            saved_epoch_num = epoch + 1
            best_val_mcc = val_mcc

        ## write loss to log_file
        with open(log_file,'a+',encoding='utf-8') as f:
            f.write("epoch_%d: loss=%.7f   tr_threshold=%.3f\n"%(epoch+1, loss.item(), mcc_threshold))
            f.write("epoch_%d: tr_TN=%d  tr_FN=%d  tr_FP=%d  tr_TP=%d  tr_Sen=%.3f  tr_Spe=%.3f  tr_Acc=%.3f  tr_MCC=%.3f  tr_AUC=%.3f\n"%(
                    epoch+1,tr_TN,tr_FN,tr_FP,tr_TP,tr_Sen,tr_Spe,tr_Acc,tr_mcc,tr_AUC))
            f.write("epoch_%d: val_TN=%d  val_FN=%d  val_FP=%d  val_TP=%d  val_Sen=%.3f  val_Spe=%.3f  val_Acc=%.3f  val_MCC=%.3f  val_AUC=%.3f\n"%(
                    epoch+1,val_TN,val_FN,val_FP,val_TP,val_Sen,val_Spe,val_Acc,val_mcc,val_AUC))
        tensorboard_writer.add_scalar('tr_threshold',mcc_threshold,epoch)
        tensorboard_writer.add_scalar('tr_Sen',tr_Sen,epoch)
        tensorboard_writer.add_scalar('tr_Spe',tr_Spe,epoch)
        tensorboard_writer.add_scalar('tr_Acc',tr_Acc,epoch)
        tensorboard_writer.add_scalar('tr_MCC',tr_mcc,epoch)
        tensorboard_writer.add_scalar('tr_AUC',tr_AUC,epoch)
        tensorboard_writer.add_scalar('val_Sen',val_Sen,epoch)
        tensorboard_writer.add_scalar('val_Spe',val_Spe,epoch)
        tensorboard_writer.add_scalar('val_Acc',val_Acc,epoch)
        tensorboard_writer.add_scalar('val_MCC',val_mcc,epoch)
        tensorboard_writer.add_scalar('val_AUC',val_AUC,epoch)