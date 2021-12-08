#%%
from model import HybridGICN
from PropDS import PhysPropData
from torch_geometric.data import DataLoader
import torch
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import optuna
from optuna.trial import TrialState
import mlflow
#%%
EPOCHS = 200
EARLY_STOPPING_STEPS = 10
NUM_PROPS = 1
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()
#%%
def get_data(prop, BS):
    trn_dt = PhysPropData(root=f'../OPERA Properties/{prop}/', filename='train.csv')
    tst_dt = PhysPropData(root=f'../OPERA Properties/{prop}/', filename='test.csv', test=True)
    torch.manual_seed(0)
    train_loader = DataLoader(trn_dt, batch_size=BS, shuffle=True)
    test_loader = DataLoader(tst_dt, batch_size=BS)
    return train_loader, test_loader
#%%
def train(model, train_loader, optimizer, criterion, device):
    train_preds, train_truths = [[] for _ in range(NUM_PROPS)], [[] for _ in range(NUM_PROPS)] # e.g. for NUM_PROPS = 2 >> [[], []]
    rmse, train_r2 = [[] for _ in range(NUM_PROPS)], [[] for _ in range(NUM_PROPS)]
    losses = 0
    step = 0
    for _, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        LOSS = []
        LABELS = [data.y.float()]
        out = model(data.x, data.edge_index, data.edge_attr.float(), data.batch)
        for i in range(NUM_PROPS):
            LOSS.append(torch.sqrt(criterion(torch.squeeze(out[i]), LABELS[i])))
            train_preds[i].extend(out[i].cpu().detach().numpy())
            train_truths[i].extend(LABELS[i].cpu().detach().numpy())
        loss = sum(LOSS)
        losses += loss.item()
        loss.backward()
        optimizer.step()
        step += 1
    train_loss = losses/step
    for i in range(NUM_PROPS):
        train_truths[i] = np.ndarray.flatten(np.array(train_truths[i]))
        train_preds[i] = np.ndarray.flatten(np.array(train_preds[i]))
        #RMSE
        rmse[i] = np.sqrt(mean_squared_error(train_truths[i], train_preds[i]))    
        #R2
        train_r2[i] = r2_score(train_truths[i], train_preds[i])

    return rmse, train_loss, train_r2

#%%
def test(model, test_loader, criterion, device):
    model.eval()
    batch_loss = 0
    test_preds, test_truths = [[] for _ in range(NUM_PROPS)], [[] for _ in range(NUM_PROPS)]
    rmsep, test_r2 = [[] for _ in range(NUM_PROPS)], [[] for _ in range(NUM_PROPS)]
    losses = 0.0
    step = 0
    for _, data in enumerate(test_loader):
        data.to(device)
        LOSS = []
        LABELS = [data.y.float()]
        out = model(data.x, data.edge_index, data.edge_attr.float(), data.batch)
        for i in range(NUM_PROPS):
            LOSS.append(torch.sqrt(criterion(torch.squeeze(out[i]), LABELS[i])))
            test_preds[i].extend(out[i].cpu().detach().numpy())
            test_truths[i].extend(LABELS[i].cpu().detach().numpy())
        loss = sum(LOSS)
        batch_loss += loss.item()    
        losses += loss.item()
        step += 1
    test_loss = losses/step
    for i in range(NUM_PROPS):
        test_truths[i] = np.ndarray.flatten(np.array(test_truths[i]))
        test_preds[i] = np.ndarray.flatten(np.array(test_preds[i]))
        #RMSEP
        rmsep[i] = np.sqrt(mean_squared_error(test_truths[i], test_preds[i]))    
        #R2
        test_r2[i] = r2_score(test_truths[i], test_preds[i])

    return rmsep, test_loss, test_r2
#%%
ES_DICT = []
def objective(trial):
    props = ['LogP','MP','BP','WS','VP','HL','AOH','KOC','BCF','KM','KOA','BioHL']
    pr = 11
    #mlflow.set_experiment(props[0])
    with mlflow.start_run(run_name=props[pr], experiment_id=pr):
        if trial.number == 201:
            print("trial stopped")
            trial.study.stop()
        # Generate the model.
        model = HybridGICN(trial, props=NUM_PROPS).to(device)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ['Adam'])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        # Get Data
        #batch_size = trial.suggest_int("BatchSize",32,128,32)
        train_loader, test_loader = get_data(props[pr], BS=32)

        # Training of the model.
        min_loss = np.Inf
        NO_IMPROV = 0
        MODEL_LIST = []
        for epoch in range(EPOCHS):
                model.train()
                rmset, train_loss, train_r2 = train(model, train_loader, optimizer, criterion, device)
                # Validation of the model.
                model.eval()
                with torch.no_grad():
                    rmsep, test_loss, test_r2 = test(model, test_loader, criterion, device)    
                
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                #test_loss = sum(rmsep)
                trial.report(rmsep[0], epoch)
                if rmsep[0] < min_loss:
                    #torch.save(model)
                    MODEL_LIST.append(model)
                    NO_IMPROV = 0
                    min_loss = rmsep[0]
                else:
                    NO_IMPROV += 1
                mlflow.log_metric(f"{props[pr]}_RMSE", rmsep[0], step=epoch)
                mlflow.log_metric(f"{props[pr]}_R2", test_r2[0], step=epoch)
                mlflow.log_metric(f"Train_RMSE", rmset[0], step=epoch)
                mlflow.log_metric("Train_R2", train_r2[0], step=epoch)
                mlflow.log_metric("Test_Loss", test_loss, step=epoch)
                mlflow.log_metric("Train_Loss", train_loss, step=epoch)
                mlflow.log_params(trial.params)
                if NO_IMPROV == EARLY_STOPPING_STEPS:
                    print(f'Early Stopping at epoch {epoch}')
                    mlflow.log_metric('Early Stopping',epoch)
                    break
                else:
                    continue    

        mlflow.pytorch.log_model(MODEL_LIST[-1],f"{props[pr]}_Model")
        
    return rmsep[0]
    #return np.round(rmsep[0],2), np.round(rmsep[1],2), np.round(rmsep[2],2)
# %%
if __name__=="__main__":
    props = ['LogP','MP','BP','WS','VP','HL','AOH','KOC','BCF','KM','KOA','BioHL']
    study = optuna.create_study(study_name=f"{props[11]}_study", direction="minimize")
    #study = optuna.create_study(directions=["minimize","minimize","minimize"])
    study.optimize(objective)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    #
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
# %%
