import json
import os

import numpy as np
import torch
import torch.nn as nn

from maskterial.modeling.common.fcresnet import FCResNet
from maskterial.utils.argparser import parse_cls_args
from maskterial.utils.data_loader import ContrastDataloader


def calculate_class_embeddings(
    model: FCResNet,
    dataloader: ContrastDataloader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and covariance matrices of embeddings for each class.

    This function extracts embeddings from the trained model for all training data,
    then calculates the mean vector and covariance matrix for each class, which
    effectively fits a Gaussian distribution to each class's embeddings.

    Args:
        model: The model with a get_embedding method
        dataloader: Data loader containing training data and labels

    Returns:
        tuple: (loc, cov) where:
            - loc: Tensor of shape (num_classes, embedding_dim) containing mean vectors
            - cov: Tensor of shape (num_classes, embedding_dim, embedding_dim) containing covariance matrices
    """
    X_train_full = torch.tensor(dataloader.X_train).float()
    y_train_full = dataloader.y_train
    model.cpu()
    model.eval()
    with torch.no_grad():
        X_embeddings = model.get_embedding(X_train_full)

    # get the mean and covariance of the embeddings for each class
    # this is equivalent to fitting a Gaussian to the embeddings of each class
    loc = torch.stack(
        [
            torch.mean(X_embeddings[y_train_full == c], dim=0)
            for c in range(dataloader.num_classes)
        ]
    )
    cov = torch.stack(
        [
            torch.cov(X_embeddings[y_train_full == c].T)
            for c in range(dataloader.num_classes)
        ]
    )

    return loc, cov


DEVICE = "cpu"  # or "cuda" if you have a GPU, but the AMM is quite small so it should be fine on CPU

if __name__ == "__main__":
    args = parse_cls_args()

    TRAIN_SEED = args.train_seed
    SAVE_DIR = args.save_dir
    CFG_PATH = args.config
    TRAIN_IMAGE_DIR = args.train_image_dir
    TRAIN_ANNOTATION_PATH = args.train_annotation_path
    TEST_IMAGE_DIR = args.test_image_dir
    TEST_ANNOTATION_PATH = args.test_annotation_path

    with open(CFG_PATH, "r") as f:
        CONFIG = json.load(f)
        TRAIN_PARAMS = CONFIG["train_params"]
        DATA_PARAMS = CONFIG["data_params"]
        MODEL_ARCHITECTURE = CONFIG["model_arch"]

        NUM_ITER = TRAIN_PARAMS["num_iterations"]
        TEST_INTERVAL = TRAIN_PARAMS["test_interval"]
        LR = TRAIN_PARAMS["learning_rate"]
        BS = TRAIN_PARAMS["batch_size"]

    # As we are sampling the data randomly, we need to set the seed to ensure reproducibility
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    dataloader = ContrastDataloader(
        train_image_dir=TRAIN_IMAGE_DIR,
        train_annotation_path=TRAIN_ANNOTATION_PATH,
        test_image_dir=TEST_IMAGE_DIR,
        test_annotation_path=TEST_ANNOTATION_PATH,
        **DATA_PARAMS,
        verbose=True,
    )

    MODEL_ARCHITECTURE["num_classes"] = dataloader.num_classes

    if TEST_ANNOTATION_PATH is not None and TEST_IMAGE_DIR is not None:
        X_test, y_test = dataloader.get_test_data()
        X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_test = torch.tensor(y_test, dtype=torch.int64).to(DEVICE)

    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    model = FCResNet(**MODEL_ARCHITECTURE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()

    #### TRAIN LOOP ####
    best_loss, test_loss = np.inf, np.inf
    for iteration in range(NUM_ITER):

        model.train()
        optimizer.zero_grad()

        X_batch, y_batch = dataloader.get_batch(batch_size=BS)
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
        y_batch = torch.tensor(y_batch, dtype=torch.int64).to(DEVICE)

        logits = model(X_batch)
        loss = loss_function(logits, y_batch)

        loss.backward()
        optimizer.step()

        if iteration % TEST_INTERVAL == 0:
            train_loss = loss.item()
            if TEST_ANNOTATION_PATH is not None:
                model.eval()
                with torch.no_grad():
                    logits = model(X_test)
                    test_loss = loss_function(logits, y_test).item()
                if test_loss < best_loss:
                    best_loss = test_loss
            print(
                f"Train Iteration: {iteration:8} Train Loss: {train_loss:10.5f} Test Loss: {test_loss:10.5f} Best Loss: {best_loss:10.5f}",
                flush=True,
            )

    if TEST_ANNOTATION_PATH is not None:
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            test_loss = loss_function(logits, y_test).item()
        if test_loss < best_loss:
            best_loss = test_loss

    #### TRAIN LOOP END ####

    # Calculate the mean and covariance of the embeddings for the GMM
    loc, cov = calculate_class_embeddings(model, dataloader)

    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(os.path.join(SAVE_DIR, "meta_data.json"), "w") as f:
        meta_data = {
            "train_config": CONFIG,
            "test_losses": {
                "final": test_loss,
                "best": best_loss,
            },
            "train_mean": dataloader.X_train_mean.tolist(),
            "train_std": dataloader.X_train_std.tolist(),
            "train_image_dir": TRAIN_IMAGE_DIR,
            "train_annotation_path": TRAIN_ANNOTATION_PATH,
            "test_image_dir": TEST_IMAGE_DIR,
            "test_annotation_path": TEST_ANNOTATION_PATH,
        }
        json.dump(meta_data, f, indent=4)

    np.save(os.path.join(SAVE_DIR, "loc.npy"), loc)
    np.save(os.path.join(SAVE_DIR, "cov.npy"), cov)

    torch.save(
        model.state_dict(),
        os.path.join(SAVE_DIR, "model.pth"),
    )
