from pathlib import Path
import random, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from hybrid_dataset import HybridDataset
from hybrid_model import HybridModel
from patient_transforms import patient_transform

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root/"oasis_model_manifest.csv")
vol = pd.read_csv(root/"oasis_volume_features.csv")
dev = df[df.Split=="Development"].merge(vol,on="Subject").reset_index(drop=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loader(data, train):
    tf=lambda x: patient_transform(x,train=train)
    return DataLoader(HybridDataset(data,root/"oasis_roi",tf),batch_size=2,shuffle=train)

def fit(model, dl, loss_fn, params, lr, epochs):
    opt=torch.optim.AdamW(params,lr=lr)
    for _ in range(epochs):
        model.train()
        for x,f,y in dl:
            x,f,y=x.to(device),f.to(device),y.to(device)
            opt.zero_grad(); loss=loss_fn(model(x,f),y)
            loss.backward(); opt.step()

def predict(model, dl):
    model.eval(); out=[]
    with torch.no_grad():
        for x,f,_ in dl:
            p=torch.softmax(model(x.to(device),f.to(device)),1)[:,1]
            out += p.cpu().tolist()
    return out

skf=StratifiedKFold(5,shuffle=True,random_state=42)
oof=[None]*len(dev)

for fold,(tr,va) in enumerate(skf.split(dev,dev.Target),1):
    random.seed(100+fold); np.random.seed(100+fold); torch.manual_seed(100+fold)
    a,b=dev.iloc[tr].copy(),dev.iloc[va].copy()

    cols=["LeftNorm","RightNorm","Asymmetry"]
    mean,std=a[cols].mean(),a[cols].std().replace(0,1)
    a[cols]=(a[cols]-mean)/std
    b[cols]=(b[cols]-mean)/std

    model=HybridModel(root/"models/kaggle_pretrained_resnet18.pt").to(device)
    n0=(a.Target=="Normal").sum(); n1=(a.Target=="Early_Demented").sum()
    w=torch.tensor([len(a)/(2*n0),len(a)/(2*n1)],dtype=torch.float32,device=device)
    loss_fn=torch.nn.CrossEntropyLoss(weight=w)

    for p in model.parameters(): p.requires_grad=False
    for p in model.classifier.parameters(): p.requires_grad=True
    fit(model,loader(a,True),loss_fn,model.classifier.parameters(),1e-3,3)

    for p in model.encoder.layer4.parameters(): p.requires_grad=True
    params=filter(lambda p:p.requires_grad,model.parameters())
    fit(model,loader(a,True),loss_fn,params,1e-4,5)

    probs=predict(model,loader(b,False))
    for i,p in zip(va,probs): oof[i]=p

    pred=[int(p>=.5) for p in probs]
    true=(b.Target=="Early_Demented").astype(int)
    print(f"Fold {fold}: BA={balanced_accuracy_score(true,pred):.3f}")

dev["Probability"]=oof
dev.to_csv(root/"results/hybrid_oof_predictions.csv",index=False)
print("Saved 44 hybrid OOF predictions")