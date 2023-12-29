from random import randint
from src.breakout import BreakoutEnv
from torch.nn.functional import one_hot
import torch
from src.model import model, train_model, optimizer, criterion
import numpy as np


def imitation(output_path):
    env = BreakoutEnv(render_mode="human")

    obs, info = env.reset()

    imitation_info = []
    while True:
        env.render()
        # action = int(input("Seleccione la acción: 0-IZQ 1-DER 2-PARAR \n"))
        action = randint(0, 2)

        imitation_info.append((action, obs))
        obs, reward, done, truncate, info = env.step(action)
        if done:
            break

    with open(output_path, "w") as out_file:
        for line in imitation_info:
            action, obs = line
            processed_obs = ", ".join(obs.astype(str))
            processed_action = one_hot(torch.tensor(action), 3).numpy().reshape(-1)
            processed_action = ", ".join(processed_action.astype(str))

            out_file.write(f"{processed_obs} | {processed_action}\n")


def train_imitation(model, file_path):
    x_train, y_train = [], []
    with open(file_path, "r") as imitation_file:
        for line in imitation_file.read().strip().split("\n"):
            obs, action = line.split(" | ")
            x_train.append([int(value) for value in obs.split(", ")])
            y_train.append([int(value) for value in action.split(", ")])

    x_data = torch.tensor(x_train, dtype=torch.float32)
    y_data = torch.tensor(y_train, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_data, y_data),
        batch_size=1,
        shuffle=True,
    )
    train_model(model, train_loader, optimizer, criterion, 20)
    torch.save(model.state_dict(), "model.pt")


def test_imitation(model, model_path):
    env = BreakoutEnv(render_mode="human")
    model.load_state_dict(torch.load(model_path))

    obs, info = env.reset()

    while True:
        action = int(
            np.argmax(model(torch.tensor(obs, dtype=torch.float32)).detach().numpy())
        )
        obs, reward, done, truncate, info = env.step(action)
        env.render()

        if done:
            break


if __name__ == "__main__":
    # imitation("imitation_steps.txt")
    train_imitation(model, "imitation_steps.txt")
    test_imitation(model, "model.pt")
