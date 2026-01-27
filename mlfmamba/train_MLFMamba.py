import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import time
import torch
import random
import argparse
import numpy as np
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import normlize3D, ImageStretching

# import matplotlib.pyplot as plt
# from visual.visualize_map import DrawResult
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from PIL import Image
from model.MLFMamba import MLFMamba

from calflops import calculate_flops

torch.autograd.set_detect_anomaly(True)

time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png', '_mask.png'), save_single_gt_path,
                      only_vis_label=True)


def sliding_window_predict(model, data_loader, height, width, device):
    if data_loader is None:
        raise ValueError("Data loader for sliding-window inference is None.")

    pred_flat = np.zeros(height * width, dtype=np.int32)
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                patch_data, _, linear_idx = batch
            else:
                patch_data, linear_idx = batch
            patch_data = patch_data.to(device)
            linear_idx = linear_idx.long().cpu().numpy()

            logits = model(patch_data)
            seg_logits = resize(input=logits,
                                size=patch_data.shape[2:],
                                mode='bilinear',
                                align_corners=True)
            center_h = seg_logits.shape[2] // 2
            center_w = seg_logits.shape[3] // 2
            center_preds = torch.argmax(seg_logits[:, :, center_h, center_w], dim=1).cpu().numpy()

            pred_flat[linear_idx] = center_preds

    return pred_flat.reshape(height, width)


# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, default=0)
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost', type=bool, default=True)
    parser.add_argument('--patch_length', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
#seed_list = [0,1,2,3,4,5,6,7,8,9]
seed_list = [0]

num_list = [args.train_samples, args.val_samples]

dataset_index = args.dataset_index

max_epoch = args.max_epoch
learning_rate = args.lr
patch_length = args.patch_length
batch_size = args.batch_size

net_name = 'MLFMamba'

paras_dict = {'net_name': net_name, 'dataset_index': dataset_index, 'num_list': num_list,
              'lr': learning_rate, 'seed_list': seed_list, 'patch_length': patch_length,
              'batch_size': batch_size}

# 0        1         2         3        4
data_set_name_list = ['IP', 'PU', 'LongKou','Houston']
data_set_name = data_set_name_list[dataset_index]

if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    setting_name = 'tr{}val{}'.format(str(args.train_samples), str(args.val_samples)) + '_lr{}'.format(
        str(learning_rate))

    dataset_name = data_set_name

    exp_name = args.exp_name

    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder, 'train_tr{}_val{}.log'.format(num_list[0], num_list[1]))
    logger = setup_logger(name='{}'.format(dataset_name), logfile=save_log_path)
    torch.cuda.empty_cache()

    logger.info(save_folder)

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)

    height, width, channels = data.shape

    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    stretched_img = ImageStretching(data).astype(np.float32) / 255.0
    full_image_tensor = torch.from_numpy(stretched_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    label_map_for_patch = gt.astype(np.int32) - 1

    class_count = max(np.unique(gt))

    flag_list = [1, 0]  # ratio or num按比例划分：ratio:[1,0];num:[1,1]
    ratio_list = [0.1, 0.01]  # [train_ratio,val_ratio]

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    evaluator = Evaluator(num_class=class_count)

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder,
                                        "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder,
                                         'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder,
                                         'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder,
                                    'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                                       num_list,
                                                                                                       gt_reshape,
                                                                                                       class_count,
                                                                                                       flag_list[1])
        index = (train_data_index, val_data_index, test_data_index)
        _, val_label, test_label = data_load_operate.generate_image_iter(data, height, width, gt_reshape,
                                                                         index)

        # build Model

        net = MLFMamba(in_channels=channels, num_classes=class_count, hidden_dim=128)
        logger.info(paras_dict)
        logger.info(net)
        train_patch_loader, val_patch_loader, test_patch_loader = data_load_operate.build_patch_dataloaders(
            stretched_img, label_map_for_patch, index, patch_length, batch_size)
        if train_patch_loader is None:
            raise RuntimeError("Training index list is empty. Please check the sampling configuration.")
        all_index = np.arange(height * width)
        full_patch_loader = data_load_operate.build_patch_dataloader(
            stretched_img, label_map_for_patch, all_index, patch_length, batch_size, shuffle=False)

        x = full_image_tensor.clone()

        test_label = test_label.to(device)
        val_label = val_label.to(device)

        # ############################################
        # val_label = test_label
        # ############################################

        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        logger.info(optimizer)
        if record_computecost:
            net.eval()
            patch_size = patch_length * 2 + 1
            input_shape = (1, channels, patch_size, patch_size)
            flops_per_patch, macs_per_patch, params = calculate_flops(model=net, input_shape=input_shape)
            total_patches = height * width
            sliding_window_flops = flops_per_patch * total_patches
            logger.info(
                "Params:{} | FLOPs per {}x{} patch:{} | Sliding-window FLOPs ({} patches):{}".format(
                    params, patch_size, patch_size, flops_per_patch, total_patches, sliding_window_flops))

        tic1 = time.perf_counter()
        best_val_acc = 0

        for epoch in range(max_epoch):
            net.train()
            train_loss_sum = 0.0
            train_pixel_correct = 0
            train_pixel_total = 0

            for batch_idx, (patch_data, patch_label, _) in enumerate(train_patch_loader):
                patch_data = patch_data.to(device)
                patch_label = patch_label.to(device)

                optimizer.zero_grad()
                logits = net(patch_data)
                seg_logits_batch = resize(input=logits,
                                          size=patch_label.shape[1:],
                                          mode='bilinear',
                                          align_corners=True)
                loss = loss_func(seg_logits_batch, patch_label.long())
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

                with torch.no_grad():
                    preds = torch.argmax(seg_logits_batch, dim=1)
                    valid_mask = patch_label != -1
                    if valid_mask.any():
                        train_pixel_correct += torch.sum((preds == patch_label) & valid_mask).item()
                        train_pixel_total += valid_mask.sum().item()

            avg_train_loss = train_loss_sum / max(1, len(train_patch_loader))
            avg_train_acc = train_pixel_correct / train_pixel_total if train_pixel_total > 0 else 0.0
            logger.info('Epoch:{}|train_loss:{:.4f}|pixel_acc:{:.4f}'.format(epoch, avg_train_loss, avg_train_acc))
            torch.cuda.empty_cache()
            # evaluate stage
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                val_pred_map = sliding_window_predict(net, val_patch_loader, height, width, device)
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), np.expand_dims(val_pred_map, axis=0))
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info(
                    'Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA, mAcc, Kappa, mIOU, IOU,
                                                                                      Acc))
                # save weight
                if OA >= best_val_acc:
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    # torch.save(net,save_weight_path)
                    torch.save(net.state_dict(), save_weight_path)
                    # save_epoch_weight_path = os.path.join(save_folder,'{}.pth'.format(str(epoch+1)))
                    # torch.save(net.state_dict(), save_epoch_weight_path)
                if (epoch + 1) % 50 == 0:
                    save_single_predict_path = os.path.join(save_vis_folder, 'predict_{}.png'.format(str(epoch + 1)))
                    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                    full_pred_map = sliding_window_predict(net, full_patch_loader, height, width, device)
                    vis_a_image(gt, full_pred_map, save_single_predict_path, save_single_gt_path)

                # net.train()
            torch.cuda.empty_cache()

        # 记录训练时间
        train_time = time.perf_counter() - tic1
        Train_Time_ALL.append(train_time)

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        pred_test = []

        load_weight_path = save_weight_path
        net.update_params = None
        best_net = MLFMamba(in_channels=channels, num_classes=class_count, hidden_dim=128)

        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        test_evaluator = Evaluator(num_class=class_count)

        # 记录测试开始时间
        test_tic = time.perf_counter()

        with torch.no_grad():
            test_evaluator.reset()
            test_pred_map = sliding_window_predict(best_net, test_patch_loader, height, width, device)
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), np.expand_dims(test_pred_map, axis=0))
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = evaluator.Kappa()
            logger.info(
                'Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test,
                                                                              mIOU_test, IOU_test,
                                                                              Acc_test))
            full_pred_map = sliding_window_predict(best_net, full_patch_loader, height, width, device)
            vis_a_image(gt, full_pred_map, predict_save_path, gt_save_path)

        # 记录测试时间
        test_time = time.perf_counter() - test_tic
        Test_Time_ALL.append(test_time)

        # Output infors
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA: {}'.format(list(OA_ALL)))
    logger.info('List of AA: {}'.format(list(AA_ALL)))
    logger.info('List of KPP: {}'.format(list(KPP_ALL)))
    logger.info('OA= {} +- {}'.format(round(np.mean(OA_ALL) * 100, 2), round(np.std(OA_ALL) * 100, 2)))
    logger.info('AA= {} +- {}'.format(round(np.mean(AA_ALL) * 100, 2), round(np.std(AA_ALL) * 100, 2)))
    logger.info('Kpp= {} +- {}'.format(round(np.mean(KPP_ALL) * 100, 2), round(np.std(KPP_ALL) * 100, 2)))
    logger.info('Acc per class= {} +- {}'.format(
        np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2),
        np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2)
    ))

    # 确保Train_Time_ALL和Test_Time_ALL不为空
    if len(Train_Time_ALL) > 0:
        logger.info("Average training time= {} +- {}".format(
            round(np.mean(Train_Time_ALL), 2),
            round(np.std(Train_Time_ALL), 3)
        ))
    else:
        logger.info("No training time data available")

    if len(Test_Time_ALL) > 0:
        logger.info("Average testing time= {} +- {}".format(
            round(np.mean(Test_Time_ALL) * 1000, 2),
            round(np.std(Test_Time_ALL) * 1000, 3)
        ))
    else:
        logger.info("No testing time data available")

    # Output infors
    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    f = open(mean_result_path, 'w')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(
        round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
        np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                  + "\nAverage training time=" + str(
        np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(
        np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
        np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
    f.close()

    del net