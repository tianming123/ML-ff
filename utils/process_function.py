import shutil
import sys
import traceback

import numpy as np
from coverage.annotate import os
from sgdml.cli import AssistantError
from network.train import FFNetTrain
from network.predict import FFNetPredict
from constant import atomic_to_number_str_dict
from functools import partial
import time

import argparse
import os
import re



# 需要保证训练范围在数据集的子集范围内
def create_task(data_set, data_set_valid, n_train, n_valid, sigs):

    n_data, n_atoms, _ = data_set['R'].shape
    if n_data < n_train:
            raise AssistantError(
                '数据集中仅包含 {} 个数据, 无法训练 {} 条.'.format(
                    n_data, n_train
                )
            )
    # 未提供单独的验证集
    if data_set_valid is None:
        if n_data - n_train < n_valid:
            raise AssistantError(
                '数据集中仅包含 {} 个数据, 无法训练 {} 条 和 验证 {} 条.'.format(
                    n_data, n_train, n_valid
                )
            )
    else:
        valid_dataset = data_set_valid
        n_valid_data = valid_dataset['R'].shape[0]
        if n_valid_data < n_valid:
            raise AssistantError(
                '验证数据集中仅包含 {} 个数据, 无法训练 {} 条.'.format(
                    n_data, n_valid
                )
            )

    if sigs is None:
        print('超参数 sigma (length scale) 采用默认值 \'10:10:100\'.')
        sigs = list(range(10,100,10))
    task_dir = 'task_%s_%d_%d'%(
            data_set['name'].astype(str),
            n_train,
            n_valid
    )
    print("文件存储目录： {}".format(task_dir))
    if os.path.exists(task_dir):
        print('目录已存在，将会覆盖目录')
    shutil.rmtree(task_dir, ignore_errors=True)
    os.makedirs(task_dir)

    # 创建训练模型
    model_train = (FFNetTrain())
    try:
        template_task = model_train.create_task(
            data_set,
            n_train,
            data_set_valid,
            n_valid,
            sig=1,
            callback=callback
        )
    except:
        print(traceback.format_exc())
        os._exit(1)
    n_written = 0
    for sig in sigs:
        template_task['sig'] = sig
        task_file_name = 'task-train%dd-sig%04d.npz'%(
            template_task['idxs_train'].shape[0],
            np.squeeze(template_task['sig']),
        )
        task_path = os.path.join(task_dir, task_file_name)
        if os.path.isfile(task_path):
            print('任务已存在 \'{}\'.'.format(task_file_name))
        else:
            np.savez_compressed(task_path, **template_task)
            n_written += 1
    if n_written > 0:
        print(
            '写入任务 {:d}/{:d} task(s) 每批次 m={} 个训练点'.format(
                n_written, len(sigs), template_task['R_train'].shape[0]
            )
        )
    return task_dir

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def calculate_error(error, size, n, mae_n_sum, rmse_n_sum):

    err = np.abs(error)

    mae_n_sum += np.sum(err) / size
    mae = mae_n_sum / n

    rmse_n_sum += np.sum(err**2) / size
    rmse = np.sqrt(rmse_n_sum / n)

    return mae, mae_n_sum, rmse, rmse_n_sum

# 在迭代过程中保存最近的模型结果
def save_progr_callback(unconv_model, unconv_model_path=None):  # Saves current (unconverged) model during iterative training
        if unconv_model_path is None:
            print('当前路径 {}，已经存在模型结果'.format(unconv_model_path))
            os._exit(1)
        np.savez_compressed(unconv_model_path, **unconv_model)

# 模型训练与测试
def model_test(
    model_dir,
    test_dataset,
    n_test,
    overwrite,
    max_memory,
    max_processes,
    use_torch
):
    model_dir, model_file_names = model_dir
    n_models = len(model_file_names)

    n_test = 0 if n_test is None else n_test
    is_validation = n_test < 0
    is_test = n_test >= 0

    dataset = test_dataset


    F_rmse = []

    DEBUG_WRITE = False

    if DEBUG_WRITE:
        if os.path.exists('test_pred.xyz'):
            os.remove('test_pred.xyz')
        if os.path.exists('test_ref.xyz'):
            os.remove('test_ref.xyz')
        if os.path.exists('test_diff.xyz'):
            os.remove('test_diff.xyz')

    num_workers, batch_size = -1, -1
    model_train = None
    for i, model_file_name in enumerate(model_file_names):

        model_path = os.path.join(model_dir, model_file_name)
        _, model = is_file_type(model_path, 'model')


        if model['use_E']:
            e_err = model['e_err'].item()
        f_err = model['f_err'].item()

        is_model_validated = not (np.isnan(f_err['mae']) or np.isnan(f_err['rmse']))

        test_idxs = model['idxs_valid']
        if is_test:
            # 保证验证集不与测试集/训练集交叉
            excl_idxs = np.empty((0,), dtype=np.uint)
            if dataset['md5'] == model['md5_train']:
                excl_idxs = np.concatenate([excl_idxs, model['idxs_train']]).astype(
                    np.uint
                )
            if dataset['md5'] == model['md5_valid']:
                excl_idxs = np.concatenate([excl_idxs, model['idxs_valid']]).astype(np.uint)
            n_data = dataset['F'].shape[0]
            n_data_eff = n_data - len(excl_idxs)

            if (n_test == 0 and n_data_eff != 0):
                n_test = n_data_eff
                print('测试集大小： {}'.format(n_test))

            if n_test == 0 or n_data_eff == 0:
                print('测试集中不存在未使用过的数据')
                return
            elif n_data_eff < n_test:
                n_test = n_data_eff
                print('数据集中不包含足够的数据点，测试集大小降低为 {:d}.'.format(n_test))

            if 'E' in dataset:
                if model_train is None:
                    model_train = FFNetTrain(
                        max_memory=max_memory, max_processes=max_processes
                    )
                test_idxs = model_train.draw_strat_sample(
                    dataset['E'], n_test, excl_idxs=excl_idxs
                )
            else:
                test_idxs = np.delete(np.arange(n_data), excl_idxs)

        # 通过shuffle来提高误差的收敛程度
        np.random.shuffle(test_idxs)

        if DEBUG_WRITE:
            test_idxs = np.sort(test_idxs)

        z = dataset['z']
        R = dataset['R'][test_idxs, :, :]
        F = dataset['F'][test_idxs, :, :]

        if model['use_E']:
            E = dataset['E'][test_idxs]

        try:
            model_predict = FFNetPredict(
                model,
                max_memory=max_memory,
                max_processes=max_processes,
                use_torch=use_torch,
            )
        except:
            print(traceback.format_exc())
            os._exit(1)

        b_size = min(1000, len(test_idxs))

        if not use_torch:
            if num_workers == -1 or batch_size == -1:
                gps, is_from_cache = model_predict.prepare_parallel(
                    n_bulk=b_size, return_is_from_cache=True
                )
                num_workers, chunk_size, bulk_mp = (
                    model_predict.num_workers,
                    model_predict.chunk_size,
                    model_predict.bulk_mp,
                )

                sec_disp_str = 'no chunking'.format(chunk_size)
                if chunk_size != model_predict.n_train:
                    sec_disp_str = 'chunks of {:d}'.format(chunk_size)

                if num_workers == 0:
                    sec_disp_str = 'no workers / ' + sec_disp_str
                else:
                    sec_disp_str = (
                        '{:d} workers {}/ '.format(
                            num_workers, '[MP] ' if bulk_mp else ''
                        )
                        + sec_disp_str
                    )
            else:
                model_predict._set_num_workers(num_workers)
                model_predict._set_chunk_size(chunk_size)
                model_predict._set_bulk_mp(bulk_mp)

        n_atoms = z.shape[0]

        if model['use_E']:
            e_mae_sum, e_rmse_sum = 0, 0
        f_mae_sum, f_rmse_sum = 0, 0
        cos_mae_sum, cos_rmse_sum = 0, 0
        mag_mae_sum, mag_rmse_sum = 0, 0

        n_done = 0
        t = time.time()
        for b_range in batch(list(range(len(test_idxs))), b_size):

            n_done_step = len(b_range)
            n_done += n_done_step

            r = R[b_range].reshape(n_done_step, -1)
            e_pred, f_pred = model_predict.predict(r)

            # energy error
            if model['use_E']:
                e = E[b_range]
                e_mae, e_mae_sum, e_rmse, e_rmse_sum = calculate_error(
                    np.squeeze(e) - e_pred, 1, n_done, e_mae_sum, e_rmse_sum
                )

                # import matplotlib.pyplot as plt
                # plt.hist(np.squeeze(e) - e_pred)
                # plt.show()

            # force component error
            f = F[b_range].reshape(n_done_step, -1)
            f_mae, f_mae_sum, f_rmse, f_rmse_sum = calculate_error(
                f - f_pred, 3 * n_atoms, n_done, f_mae_sum, f_rmse_sum
            )

            # magnitude error
            f_pred_mags = np.linalg.norm(f_pred.reshape(-1, 3), axis=1)
            f_mags = np.linalg.norm(f.reshape(-1, 3), axis=1)
            mag_mae, mag_mae_sum, mag_rmse, mag_rmse_sum = calculate_error(
                f_pred_mags - f_mags, n_atoms, n_done, mag_mae_sum, mag_rmse_sum
            )

            # normalized cosine error
            f_pred_norm = f_pred.reshape(-1, 3) / f_pred_mags[:, None]
            f_norm = f.reshape(-1, 3) / f_mags[:, None]
            cos_err = (
                np.arccos(np.clip(np.einsum('ij,ij->i', f_pred_norm, f_norm), -1, 1))
                / np.pi
            )
            cos_mae, cos_mae_sum, cos_rmse, cos_rmse_sum = calculate_error(
                cos_err, n_atoms, n_done, cos_mae_sum, cos_rmse_sum
            )

            # NEW

            if is_test and DEBUG_WRITE:

                try:
                    with open('test_pred.xyz', 'a') as file:

                        n = r.shape[0]
                        for i, ri in enumerate(r):

                            r_out = ri.reshape(-1, 3)
                            e_out = e_pred[i]
                            f_out = f_pred[i].reshape(-1, 3)

                            ext_xyz_str = (
                                generate_xyz_str(r_out, model['z'], e=e_out, f=f_out)
                                + '\n'
                            )

                            file.write(ext_xyz_str)

                except IOError:
                    sys.exit("文件写入失败")

                try:
                    with open('test_ref.xyz', 'a') as file:

                        n = r.shape[0]
                        for i, ri in enumerate(r):

                            r_out = ri.reshape(-1, 3)
                            e_out = (
                                None
                                if not model['use_E']
                                else np.squeeze(E[b_range][i])
                            )
                            f_out = f[i].reshape(-1, 3)

                            ext_xyz_str = (
                                generate_xyz_str(r_out, model['z'], e=e_out, f=f_out)
                                + '\n'
                            )
                            file.write(ext_xyz_str)

                except IOError:
                    sys.exit("文件写入失败")

                try:
                    with open('test_diff.xyz', 'a') as file:

                        n = r.shape[0]
                        for i, ri in enumerate(r):

                            r_out = ri.reshape(-1, 3)
                            e_out = (
                                None
                                if not model['use_E']
                                else (np.squeeze(E[b_range][i]) - e_pred[i])
                            )
                            f_out = (f[i] - f_pred[i]).reshape(-1, 3)

                            ext_xyz_str = (
                                generate_xyz_str(r_out, model['z'], e=e_out, f=f_out)
                                + '\n'
                            )
                            file.write(ext_xyz_str)

                except IOError:
                    sys.exit("文件写入失败")

            sps = n_done / (time.time() - t)  # examples per second
            disp_str = 'energy %.3f/%.3f, ' % (e_mae, e_rmse) if model['use_E'] else ''
            disp_str += 'forces %.3f/%.3f' % (f_mae, f_rmse)
            disp_str = (
                '{} errors (MAE/RMSE): '.format('Test' if is_test else 'Validation')
                + disp_str
            )
            sec_disp_str = '@ %.1f geo/s' % sps if b_range is not None else ''


        if model['use_E']:
            e_rmse_pct = (e_rmse / e_err['rmse'] - 1.0) * 100
        f_rmse_pct = (f_rmse / f_err['rmse'] - 1.0) * 100

        if is_test and n_models == 1:
            n_train = len(model['idxs_train'])
            n_valid = len(model['idxs_valid'])
            print()

            r_unit = 'unknown unit'
            e_unit = 'unknown unit'
            f_unit = 'unknown unit'
            if 'r_unit' in dataset and 'e_unit' in dataset:
                r_unit = dataset['r_unit']
                e_unit = dataset['e_unit']
                f_unit = str(dataset['e_unit']) + '/' + str(dataset['r_unit'])

            format_str = '  {:<18} {:>.4f}/{:>.4f} [{}]'
            if model['use_E']:
                print('Energy:  MAE: %s ,RMSE: %s' % (e_mae,e_rmse))
            print('Force:  MAE: %s ,RMSE: %s' % (f_mae,f_rmse))
            print(format_str.format('  Magnitude:', mag_mae, mag_rmse, r_unit))
            print(format_str.format('  Angle:', cos_mae, cos_rmse, '0-1'))
            print()

        model_mutable = dict(model)
        model.close()
        model = model_mutable

        model_needs_update = (
            overwrite
            or (is_test and model['n_test'] < len(test_idxs))
            or (is_validation and not is_model_validated)
        )
        if model_needs_update:

            if is_validation and overwrite:
                model['n_test'] = 0  # flag the model as not tested

            if is_test:
                model['n_test'] = len(test_idxs)
                model['md5_test'] = dataset['md5']

            if model['use_E']:
                model['e_err'] = {
                    'mae': e_mae.item(),
                    'rmse': e_rmse.item(),
                }

            model['f_err'] = {'mae': f_mae.item(), 'rmse': f_rmse.item()}
            np.savez_compressed(model_path, **model)

            if is_test and model['n_test'] > 0:
                print('Expected errors were updated in model file.')

        else:
            add_info_str = (
                'the same number of'
                if model['n_test'] == len(test_idxs)
                else 'only {:,}'.format(len(test_idxs))
            )

        F_rmse.append(f_rmse)

    return F_rmse

def model_train(
    task_dir,
    data_set,
    lazy_training,
    overwrite,
    max_memory,
    max_processes,
    use_torch,
    command=None,
    **kwargs
):

    valid_dataset = data_set
    task_dir, task_file_names = task_dir
    n_tasks = len(task_file_names)

    #创建模型
    try:
        train_model = FFNetTrain(
            max_memory=max_memory, max_processes=max_processes, use_torch=use_torch
        )
    except:
        print("训练模型创建失败！")
        print(traceback.format_exc())
        os._exit(1)

    prev_valid_err = -1
    has_converged_once = False

    for i, task_file_name in enumerate(task_file_names):
        task_file_path = os.path.join(task_dir, task_file_name)
        with np.load(task_file_path, allow_pickle=True) as task:

            if n_tasks > 1:
                if i > 0:
                    print()

                n_train = len(task['idxs_train'])
                n_valid = len(task['idxs_valid'])

                print('Task {:d} of {:d}'.format(i + 1, n_tasks),
                    '{:,} + {:,} points (training + validation), sigma (length scale): {}'.format(
                        n_train, n_valid, task['sig']
                    ),
                )

            model_file_name = model_file_name_func(task, is_extended=False)
            model_file_path = os.path.join(task_dir, model_file_name)

            if not overwrite and os.path.isfile(model_file_path):
                print('模型 \'{}\' 已经存在.'.format(model_file_name))

                model_path = os.path.join(task_dir, model_file_name)
                _, model = is_file_type(model_path, 'model')

                energy_err = {'mae': 0.0, 'rmse': 0.0}
                if model['use_E']:
                    energy_err = model['e_err'].item()
                force_err = model['f_err'].item()

                is_conv = True
                if 'solver_resid' in model:
                    is_conv = (
                            model['solver_resid']
                            <= model['solver_tol'] * model['norm_y_train']
                    )

                is_model_validated = not (
                        np.isnan(force_err['mae']) or np.isnan(force_err['rmse'])
                )
                if is_model_validated:
                    disp_str = (
                        'energy %.3f/%.3f, ' % (energy_err['mae'], energy_err['rmse'])
                        if model['use_E']
                        else ''
                    )
                    disp_str += 'forces %.3f/%.3f' % (force_err['mae'], force_err['rmse'])
                    disp_str = 'Validation errors (MAE/RMSE): ' + disp_str
                    print(disp_str)
                    valid_errs = [force_err['rmse']]

            else:  # 训练验证模型

                if lazy_training and n_tasks > 1:
                    if 'tried_training' in task and task['tried_training']:
                        print('任务存在失败记录，跳过该任务')
                        continue

                #记录任务文件
                task = dict(task)
                task['tried_training'] = True
                np.savez_compressed(task_file_path, **task)

                n_train, n_atoms = task['R_train'].shape[:2]

                unconv_model_file = '_unconv_{}'.format(model_file_name)
                unconv_model_path = os.path.join(task_dir, unconv_model_file)

                try:
                    model = train_model.train(
                        task,
                        partial(save_progr_callback, unconv_model_path=unconv_model_path),
                    )
                except:
                    print(traceback.format_exc())
                    os._exit(1)
                else:
                    np.savez_compressed(model_file_path, **model)

                    # 覆盖式写入文件目录
                    unconv_model_exists = os.path.isfile(unconv_model_path)
                    if unconv_model_exists:
                        os.remove(unconv_model_path)

                is_model_validated = False
            if not is_model_validated:

                # 当任务数大于1时，开启验证环节
                if (n_tasks == 1):
                    print('跳过验证步骤')
                    break

                # 验证模型结果
                model_dir = (task_dir, [model_file_name])
                valid_errs = model_test(
                    model_dir,
                    valid_dataset,
                    -1,
                    max_memory,
                    max_processes,
                    use_torch,
                    command,
                    **kwargs
                )

                is_conv = True
                if 'solver_resid' in model:
                    is_conv = (
                            model['solver_resid']
                            <= model['solver_tol'] * model['norm_y_train']
                    )
            has_converged_once = has_converged_once or is_conv
            # 拐点，训练损失降低，验证损失提高
            # if (
            #         has_converged_once
            #         and prev_valid_err != -1
            #         and prev_valid_err < valid_errs[0]
            # ):
            #     print('验证损失再次升高，提前结束剩余训练任务')
            #     break

            prev_valid_err = valid_errs[0]
    model_dir_or_file_path = model_file_path if n_tasks == 1 else task_dir

    return model_dir_or_file_path

# 超参数选择
def hyper_parameter_select(model_dir, overwrite, model_file=None, command=None, **kwargs):

    any_model_not_validated = False
    any_model_is_tested = False

    model_dir, model_file_names = model_dir
    if len(model_file_names) > 1:

        use_E = True

        rows = []
        data_names = ['sig', 'MAE', 'RMSE', 'MAE', 'RMSE']
        for i, model_file_name in enumerate(model_file_names):
            model_path = os.path.join(model_dir, model_file_name)
            _, model = is_file_type(model_path, 'model')

            use_E = model['use_E']

            if i == 0:
                idxs_train = set(model['idxs_train'])
                md5_train = model['md5_train']
                idxs_valid = set(model['idxs_valid'])
                md5_valid = model['md5_valid']
            else:
                if (
                    md5_train != model['md5_train']
                    or md5_valid != model['md5_valid']
                    or idxs_train != set(model['idxs_train'])
                    or idxs_valid != set(model['idxs_valid'])
                ):
                    raise AssistantError("数据集前后未对齐")


            e_err = {'mae': 0.0, 'rmse': 0.0}
            if model['use_E']:
                e_err = model['e_err'].item()
            f_err = model['f_err'].item()

            is_model_validated = not (np.isnan(f_err['mae']) or np.isnan(f_err['rmse']))
            if not is_model_validated:
                any_model_not_validated = True

            is_model_tested = model['n_test'] > 0
            if is_model_tested:
                any_model_is_tested = True

            rows.append(
                [model['sig'], e_err['mae'], e_err['rmse'], f_err['mae'], f_err['rmse']]
            )

            model.close()



        f_rmse_col = [row[4] for row in rows]
        best_idx = f_rmse_col.index(min(f_rmse_col))
        best_sig = rows[best_idx][0]

        rows = sorted(rows, key=lambda col: col[0])
        print('交叉验证误差：')
        print(' ' * 7 + 'Energy' + ' ' * 6 + 'Forces')
        print((' {:>3} ' + '{:>5} ' * 4).format(*data_names))
        print(' ' + '-' * 27)
        format_str = ' {:>3} ' + '{:5.2f} ' * 4
        format_str_no_E = ' {:>3}     -     - ' + '{:5.2f} ' * 2
        for row in rows:
            if use_E:
                row_str = format_str.format(*row)
            else:
                row_str = format_str_no_E.format(*[row[0], row[3], row[4]])
            print(row_str)
        print()

        sig_col = [row[0] for row in rows]

    else:
        print('目录下仅有一个模型')

        best_idx = 0

    best_model_path = os.path.join(model_dir, model_file_names[best_idx])

    if model_file is None:

        best_model = np.load(best_model_path, allow_pickle=True)
        model_file = model_file_name_func(best_model, is_extended=True)
        best_model.close()

    model_exists = os.path.isfile(model_file)
    if model_exists and overwrite:
        print('覆盖式写入模型文件')

    if not model_exists or overwrite:

        shutil.copy(best_model_path, model_file)
        shutil.rmtree(model_dir, ignore_errors=True)

    return model_file

# 检查目录包含指定类型的文件
def check_dir_with_specified_type(arg, type, or_file=False):
    if or_file and os.path.isfile(arg):  # arg: file path
        _, file = is_file_type(
            arg, type
        )  # raises exception if there is a problem with the file
        file.close()
        file_name = os.path.basename(arg)
        file_dir = os.path.dirname(arg)
        return file_dir, [file_name]
    else:  # arg: dir

        if not os.path.isdir(arg):
            raise argparse.ArgumentTypeError('{0} is not a directory'.format(arg))

        file_names = filter_file_type(arg, type)

        # if not len(file_names):
        #    raise argparse.ArgumentTypeError(
        #        '{0} contains no {1} files'.format(arg, type)
        #    )

        return arg, file_names

# 检查文件类型
def is_file_type(arg, type):
    # Replace MD5 dataset fingerprint with file name, if necessary.
    if type == 'dataset' and not arg.endswith('.npz') and not os.path.isdir(arg):
        dir = '.'
        if re.search(r'^[a-f0-9]{32}$', arg):  # arg looks similar to MD5 hash string
            md5_str = arg
        else:  # is it a path with a MD5 hash at the end?
            md5_str = os.path.basename(os.path.normpath(arg))
            dir = os.path.dirname(os.path.normpath(arg))

            if dir == '':  # it is only a filename after all, hence not the right type
                raise argparse.ArgumentTypeError('{0} is not a .npz file'.format(arg))

            if re.search(r'^[a-f0-9]{32}$', md5_str) and not os.path.isdir(
                dir
            ):  # path has MD5 hash string at the end, but directory is not valid
                raise argparse.ArgumentTypeError('{0} is not a directory'.format(dir))

        file_names = filter_file_type(dir, type, md5_match=md5_str)

        if not len(file_names):
            raise argparse.ArgumentTypeError(
                "No {0} files with fingerprint '{1}' found in '{2}'".format(
                    type, md5_str, dir
                )
            )
        elif len(file_names) > 1:
            error_str = (
                "Multiple {0} files with fingerprint '{1}' found in '{2}'".format(
                    type, md5_str, dir
                )
            )
            for file_name in file_names:
                error_str += '\n       {0}'.format(file_name)

            raise argparse.ArgumentTypeError(error_str)
        else:
            arg = os.path.join(dir, file_names[0])

    if not arg.endswith('.npz'):
        argparse.ArgumentTypeError('{0} is not a .npz file'.format(arg))

    try:
        file = np.load(arg, allow_pickle=True)
    except Exception:
        raise argparse.ArgumentTypeError('{0} is not readable'.format(arg))

    if 'type' not in file or file['type'].astype(str) != type[0]:
        raise argparse.ArgumentTypeError('{0} is not a {1} file'.format(arg, type))

    return arg, file

# 筛选指定类型的文件
def filter_file_type(dir, type, md5_match=None):

    file_names = []
    for file_name in sorted(os.listdir(dir)):
        if file_name.endswith('.npz'):
            file_path = os.path.join(dir, file_name)
            try:
                file = np.load(file_path, allow_pickle=True)
            except Exception:
                raise argparse.ArgumentTypeError(
                    'contains unreadable .npz files'
                )

            if 'type' in file and file['type'].astype(str) == type[0]:

                if md5_match is None:
                    file_names.append(file_name)
                elif 'md5' in file and file['md5'] == md5_match:
                    file_names.append(file_name)

            file.close()

    return file_names

def model_file_name_func(task_or_model, is_extended=False):

    n_train = task_or_model['idxs_train'].shape[0]
    n_perms = task_or_model['perms'].shape[0]
    sig = np.squeeze(task_or_model['sig'])

    if is_extended:
        dataset = np.squeeze(task_or_model['dataset_name'])
        theory_level_str = re.sub(
            r'[^\w\-_\.]', '.', str(np.squeeze(task_or_model['dataset_theory']))
        )
        theory_level_str = re.sub(r'\.\.', '.', theory_level_str)
        return '%s-%s-train%d-sym%d.npz' % (dataset, theory_level_str, n_train, n_perms)
    return 'model-train%d-sym%d-sig%04d.npz' % (n_train, n_perms, sig)
# 生成目标类型
def generate_xyz_str(r, z, e=None, f=None, lattice=None):

    comment_str = ''
    if lattice is not None:
        comment_str += 'Lattice=\"{}\" '.format(
            ' '.join(['{:.12g}'.format(l) for l in lattice.T.ravel()])
        )
    if e is not None:
        comment_str += 'Energy={:.12g} '.format(e)
    comment_str += 'Properties=species:S:1:pos:R:3'
    if f is not None:
        comment_str += ':forces:R:3'

    species_str = '\n'.join([atomic_to_number_str_dict[z_i] for z_i in z])

    r_f_str = gen_mat_str(r)[0]
    if f is not None:
        r_f_str = merge_col_str(r_f_str, gen_mat_str(f)[0])

    xyz_str = str(len(r)) + '\n' + comment_str + '\n'
    xyz_str += merge_col_str(species_str, r_f_str)

    return xyz_str

# 合并两个多行字符串
def merge_col_str(
    col_str1, col_str2
):
    return '\n'.join(
        [
            ' '.join([c1, c2])
            for c1, c2 in zip(col_str1.split('\n'), col_str2.split('\n'))
        ]
    )

# 将矩阵转换为字符串
def gen_mat_str(mat):


    def _int_len(
        x,
    ):  # length of string representation before decimal point (including sign)
        return len(str(int(abs(x)))) + (0 if x >= 0 else 1)

    def _dec_len(x):  # length of string representation after decimal point

        x_str_split = '{:g}'.format(x).split('.')
        return len(x_str_split[1]) if len(x_str_split) > 1 else 0

    def _max_int_len_for_col(
        mat, col
    ):  # length of string representation before decimal point for each col
        col_min = np.min(mat[:, col])
        col_max = np.max(mat[:, col])
        return max(_int_len(col_min), _int_len(col_max))

    def _max_dec_len_for_col(
        mat, col
    ):  # length of string representation after decimal point for each col
        return max([_dec_len(cell) for cell in mat[:, col]])

    n_cols = mat.shape[1]
    col_int_widths = [_max_int_len_for_col(mat, i) for i in range(n_cols)]
    col_dec_widths = [_max_dec_len_for_col(mat, i) for i in range(n_cols)]
    col_widths = [iw + cd + 1 for iw, cd in zip(col_int_widths, col_dec_widths)]

    mat_str = ''
    for row in mat:
        if mat_str != '':
            mat_str += '\n'
        mat_str += ' '.join(
            ' ' * max(col_int_widths[j] - _int_len(x), 0)
            + ('{: <' + str(_int_len(x) + col_dec_widths[j] + 1) + 'g}').format(x)
            for j, x in enumerate(row)
        )

    return mat_str, col_widths

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GRAY = list(range(8)) + [60]
COLOR_SEQ, RESET_SEQ = '\033[{:d};{:d};{:d}m', '\033[0m'
MAX_PRINT_WIDTH = 100
LOG_LEVELNAME_WIDTH = 7  # do not modify
ENABLE_COLORED_OUTPUT = (
    sys.stdout.isatty()
)  # Running in a real terminal or piped/redirected?

def callback(
    current,
    total=1,
    disp_str='',
    sec_disp_str=None,
    done_with_warning=False,
    newline_when_done=True,
):
    """
    Print progress or toggle bar.

    Example (progress):
    ``[ 45%] Task description (secondary string)``

    Example (toggle, not done):
    ``[ .. ] Task description (secondary string)``

    Example (toggle, done):
    ``[DONE] Task description (secondary string)``

    Parameters
    ----------
        current : int
            How many items already processed?
        total : int, optional
            Total number of items? If there is only
            one item, the toggle style is used.
        disp_str : :obj:`str`, optional
            Task description.
        sec_disp_str : :obj:`str`, optional
            Additional string shown in gray.
        done_with_warning : bool, optional
            Indicate that the process did not
            finish successfully.
        newline_when_done : bool, optional
            Finish with a newline character once
            current=total (default: True)?
    """

    global last_callback_pct

    is_toggle = total == 1
    is_done = np.isclose(current - total, 0.0)

    bold_color_str = partial(color_str, bold=True)

    if is_toggle:

        if is_done:
            if done_with_warning:
                flag_str = bold_color_str('[WARN]', fore_color=YELLOW)
            else:
                flag_str = bold_color_str('[DONE]', fore_color=GREEN)

        else:
            flag_str = bold_color_str('[' + blink_str(' .. ') + ']')
    else:

        # Only show progress in 10 percent steps when not printing to terminal.
        pct = int(float(current) * 100 / total)
        pct = int(np.ceil(pct / 10.0)) * 10 if not sys.stdout.isatty() else pct

        # Do not print, if there is no need to.


        last_callback_pct = pct

        flag_str = bold_color_str(
            '[{:3d}%]'.format(pct), fore_color=GREEN if is_done else WHITE
        )

    sys.stdout.write('\r{} {}'.format(flag_str, disp_str))

    if sec_disp_str is not None:
        w = MAX_PRINT_WIDTH - LOG_LEVELNAME_WIDTH - len(disp_str) - 1
        # sys.stdout.write(' \x1b[90m{0: >{width}}\x1b[0m'.format(sec_disp_str, width=w))
        sys.stdout.write(
            color_str(' {:>{width}}'.format(sec_disp_str, width=w), fore_color=GRAY)
        )

    if is_done and newline_when_done:
        sys.stdout.write('\n')

    sys.stdout.flush()
def color_str(str, fore_color=WHITE, back_color=BLACK, bold=False):

    if ENABLE_COLORED_OUTPUT:

        # foreground is set with 30 plus the number of the color, background with 40
        return (
            COLOR_SEQ.format(1 if bold else 0, 30 + fore_color, 40 + back_color)
            + str
            + RESET_SEQ
        )
    else:
        return str
def blink_str(str):

    return '\x1b[5m' + str + '\x1b[0m' if ENABLE_COLORED_OUTPUT else str