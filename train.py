import os
import torch
from tqdm import tqdm


def train_loop_one_image(model, train_loader, val_loader, loss_f, optimizer, epochs, device, writer, path_save, scheduler):

    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        correct_predictions = 0
        total_labels = 0
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for i, (img1, label1, path1) in enumerate(train_tqdm):
            img1, label1 = img1.to(device), label1.to(device)
            labels = label1
            optimizer.zero_grad()
            outputs = model(img1)
            loss = loss_f(outputs, label1)
            loss.backward()
            optimizer.step()

            # Save values
            avg_loss += loss.item()
            _, predicted = outputs.max(1)
            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()
            curr_loss = loss.item()
            curr_acc = 100. * correct_predictions / total_labels

            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Iteration_Loss", curr_loss, global_step)
            writer.add_scalar("Train/Iteration_Acc", curr_acc, global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)

            train_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })

        final_loss = avg_loss / len(train_loader)
        final_accuracy = 100. * correct_predictions / total_labels

        writer.add_scalar("Train/Epoch_Loss", final_loss, epoch+1)
        writer.add_scalar("Train/Epoch_Acc", final_accuracy, epoch+1)

        if (epoch+1) % 3 == 0:
            eval_model_one_image(model, val_loader, loss_f, device, epoch, writer)
            torch.save(model.state_dict(), os.path.join(path_save, 'model_epoch{}.pt'.format(epoch+1)))

        if scheduler is not None:
            scheduler.step()

    writer.flush()


def eval_model_one_image(model, val_loader, loss_f, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_labels = 0
    val_tqdm = tqdm(val_loader, desc=f"Validating Model at Epoch {epoch+1}", leave=False)
    with torch.no_grad():
        for i, (img1, label1, path1) in enumerate(val_tqdm):
            img1, label1 = img1.to(device), label1.to(device)
            labels = label1

            outputs = model(img1)
            loss = loss_f(outputs, label1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_labels += batch_size
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()

            curr_acc = 100. * correct_predictions / total_labels
            val_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })

    final_loss = total_loss / total_labels
    final_accuracy = 100. * correct_predictions / total_labels

    writer.add_scalar("Val/Loss", final_loss, epoch+1)
    writer.add_scalar("Val/Acc", final_accuracy, epoch+1)


def train_loop(model, train_loader, val_loader, loss_f, optimizer, epochs, device, writer, path_save, scheduler):
    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        correct_predictions = 0
        total_labels = 0
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for i, (img1, img2, label1, label2, path1, path2) in enumerate(train_tqdm):
            img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)

            assert list(label1) == list(label2)
            labels = label1

            optimizer.zero_grad()
            outputs, _ = model(img1, img2)
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # Save values
            avg_loss += loss.item()
            _, predicted = outputs.max(1)
            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()
            curr_loss = loss.item()
            curr_acc = 100. * correct_predictions / total_labels
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Train/Iteration_Loss", curr_loss, global_step)
            writer.add_scalar("Train/Iteration_Acc", curr_acc, global_step)
            writer.add_scalar("Train/LR", optimizer.param_groups[0]['lr'], global_step)

            train_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })

        final_loss = avg_loss / len(train_loader)
        final_accuracy = 100. * correct_predictions / total_labels

        writer.add_scalar("Train/Epoch_Loss", final_loss, epoch+1)
        writer.add_scalar("Train/Epoch_Acc", final_accuracy, epoch+1)

        if (epoch+1) % 3 == 0:
            eval_model(model, val_loader, loss_f, device, epoch, writer)
            torch.save(model.state_dict(), os.path.join(path_save, 'model_epoch{}.pt'.format(epoch+1)))

        if scheduler is not None:
            scheduler.step()

    writer.flush()


def eval_model(model, val_loader, loss_f, device, epoch, writer):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_labels = 0
    val_tqdm = tqdm(val_loader, desc=f"Validating Model at Epoch {epoch+1}", leave=False)
    with torch.no_grad():
        for i, (img1, img2, label1, label2, path1, path2) in enumerate(val_tqdm):
            img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)

            assert list(label1) == list(label2)
            assert os.path.basename(path1[0]).split('_')[0] == os.path.basename(path2[0]).split('_')[0]

            labels = label1

            outputs, _ = model(img1, img2)
            loss = loss_f(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_labels += batch_size
            _, predicted = outputs.max(1)

            correct_predictions += predicted.eq(labels).sum().item()

            curr_acc = 100. * correct_predictions / total_labels
            val_tqdm.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{curr_acc:.2f}'
            })

    final_loss = total_loss / total_labels
    final_accuracy = 100. * correct_predictions / total_labels

    writer.add_scalar("Val/Loss", final_loss, epoch+1)
    writer.add_scalar("Val/Acc", final_accuracy, epoch+1)


def test_model_one_image(model, test_loader, device):
    model.eval()
    all_pred = []
    all_labels = []
    all_pred_prob = []
    all_paths1 = []
    all_names1 = []
    embeddings = []
    correct_predictions = 0
    total_labels = 0
    test_tqdm = tqdm(test_loader, desc="Testing Model:", leave=False)
    with torch.no_grad():
        for i, (img1, label1, path1) in enumerate(test_tqdm):
            img1, label1 = img1.to(device), label1.to(device)
            labels = label1

            outputs_logits = model(img1)
            outputs = torch.softmax(outputs_logits, dim=1)
            embedding = model.forward_features(img1)
            embedding = model.forward_head(embedding, pre_logits=True)

            # Save values
            probs, predicted = outputs.max(1)

            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

            all_pred_prob.append(outputs[:, 1])
            all_pred.append(predicted)
            all_labels.append(labels)
            all_paths1.extend(path1)
            embeddings.append(embedding)

            all_names1.extend(os.path.basename(p) for p in path1)
            curr_acc = 100. * correct_predictions / total_labels
            test_tqdm.set_postfix({
                'Acc': f'{curr_acc:.2f}'
            })

    return all_pred, all_labels, all_pred_prob, all_paths1, correct_predictions, total_labels, all_names1, embeddings


def test_model_gradCam(model, test_loader, device):
    model.eval()
    all_pred = []
    all_labels = []
    all_pred_prob = []
    all_paths1 = []
    all_paths2 = []
    all_names1 = []
    all_names2 = []
    embeddings = []
    correct_predictions = 0
    total_labels = 0
    test_tqdm = tqdm(test_loader, desc="Testing Model:", leave=False)
    with torch.no_grad():
        for i, (img1, img2, label1, label2, path1, path2) in enumerate(test_tqdm):
            img1, img2, label1, label2 = img1.to(device), img2.to(device), label1.to(device), label2.to(device)
            assert list(label1) == list(label2)
            assert os.path.basename(path1[0]).split('_')[0] == os.path.basename(path2[0]).split('_')[0]

            labels = label1
            outputs_logits, embedding = model(img1, img2)
            outputs = torch.softmax(outputs_logits, dim=1)

            # Save values
            probs, predicted = outputs.max(1)

            total_labels += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

            all_pred_prob.append(outputs[:, 1])
            all_pred.append(predicted)
            all_labels.append(labels)
            all_paths1.extend(path1)
            all_paths2.extend(path2)
            all_names1.extend(os.path.basename(p) for p in path1)
            all_names2.extend(os.path.basename(p) for p in path2)
            embeddings.append(embedding)

            curr_acc = 100. * correct_predictions / total_labels
            test_tqdm.set_postfix({
                'Acc': f'{curr_acc:.2f}'
            })

    return all_pred, all_labels, all_pred_prob, all_paths1, all_paths2, correct_predictions, total_labels, all_names1, all_names2, embeddings
