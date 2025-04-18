import pdb

import torch


def train(
    model,
    dataloader,
    optimizer,
    data,
    is_cuda=False,
    normalize_input=True,
    ouput_diff=True,
    normalize_ouput=True,
    dic_norm={},
    learn_twod_output=True,
):
    loss_list = []
    relative_error_list = []
    for batch_index, graph in enumerate(dataloader):
        # graph = transformer(graph)
        if is_cuda:
            graph = graph.cuda()

        crds = graph.pos
        if data == "deforming_plate":
            input_frame = graph.x[:, -1]
            target = graph.y
        else:
            if learn_twod_output:
                input_frame = torch.vstack([graph.x[:, -1], graph.x[:, -2]]).transpose(
                    1, 0
                )
                target = graph.y
            else:
                input_frame = graph.x[:, -1]
                input_frame = input_frame.reshape(input_frame.shape[0], 1)
                target = graph.y[:, -1]
                target = target.reshape(target.shape[0], 1)

        if ouput_diff:
            target = target - input_frame

        if len(dic_norm) == 0:
            mean_input = torch.mean(input_frame)
            std_input = torch.std(input_frame)
            mean_output = torch.mean(target)
            std_output = torch.std(target)
        else:
            mean_input = dic_norm["mean_input"]
            std_input = dic_norm["std_input"]
            mean_output = dic_norm["mean_output"]
            std_output = dic_norm["std_output"]

        if normalize_input:
            input_frame = (input_frame - mean_input) / std_input

        if normalize_ouput:
            target = (target - mean_output) / std_output

        # node_type = graph.x[:, 0]  # "node_type, cur_v, pressure, time"
        # velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc = model(crds, input_frame)
        if normalize_input and not normalize_ouput:
            predicted_acc = predicted_acc * std_input + mean_input

        errors = (predicted_acc - target) ** 2
        loss = torch.mean(errors)
        relative_error = torch.mean(errors) / torch.mean((target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if normalize_ouput:
            loss = torch.mean((predicted_acc * std_input - target * std_input) ** 2)

        loss_list.append(loss.detach())
        relative_error_list.append(relative_error.detach())

    return model, loss_list, relative_error_list, target.detach(), input_frame.detach()


def test(
    model,
    dataloader,
    data,
    is_cuda=False,
    normalize_input=True,
    ouput_diff=True,
    normalize_ouput=True,
    dic_norm={},
    learn_twod_output=True,
):
    loss_list = []
    relative_error_list = []
    for batch_index, graph in enumerate(dataloader):
        # graph = transformer(graph)
        if is_cuda:
            graph = graph.cuda()

        crds = graph.pos
        if data == "deforming_plate":
            input_frame = graph.x[:, -1]
            target = graph.y
        else:
            if learn_twod_output:
                input_frame = torch.vstack([graph.x[:, -1], graph.x[:, -2]]).transpose(
                    1, 0
                )
                target = graph.y
            else:
                input_frame = graph.x[:, -1]
                input_frame = input_frame.reshape(input_frame.shape[0], 1)
                target = graph.y[:, -1]
                target = target.reshape(target.shape[0], 1)

        if ouput_diff:
            target = target - input_frame

        if len(dic_norm) == 0:
            mean_input = torch.mean(input_frame)
            std_input = torch.std(input_frame)
            mean_output = torch.mean(target)
            std_output = torch.std(target)
        else:
            mean_input = dic_norm["mean_input"]
            std_input = dic_norm["std_input"]
            mean_output = dic_norm["mean_output"]
            std_output = dic_norm["std_output"]

        if normalize_input:
            input_frame = (input_frame - mean_input) / std_input

        if normalize_ouput:
            target = (target - mean_output) / std_output
        # node_type = graph.x[:, 0]  # "node_type, cur_v, pressure, time"
        # velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc = model(crds, input_frame)
        if normalize_input and not normalize_ouput:
            predicted_acc = predicted_acc * std_input + mean_input

        errors = (predicted_acc - target) ** 2
        loss = torch.mean(errors)
        relative_error = torch.mean(errors) / torch.mean((target) ** 2)
        if normalize_ouput:
            loss = torch.mean((predicted_acc * std_output - target * std_output) ** 2)
        loss_list.append(loss.detach())
        relative_error_list.append(relative_error.detach())

    return loss_list, relative_error_list, target.detach(), input_frame.detach()
