from utils.process_function import *
import psutil
import multiprocessing as mp

file_path = "md17_ethanol.npz"
file = np.load(file_path)

n_train = 200
n_test = 200
n_validate = 200
sigs = list(range(10, 100, 10))  # 超参数sigma
lazy_training = False
overwrite = True
use_torch = True
max_memory = None
max_processes = None
model_file = None
data_set = file
test_data_set = None
valid_data_set = None

if __name__ == '__main__':

    total_memory = psutil.virtual_memory().total // 2 ** 30
    total_cpus = mp.cpu_count()

    if max_memory is None:
        max_memory = total_memory
    if max_processes is None:
        max_processes = total_cpus

    if test_data_set is None:
        test_data_set = data_set
    if valid_data_set is None:
        valid_data_set = data_set


    task_dir = create_task(data_set,valid_data_set,n_train,n_validate,sigs)
    task_dir_arg = check_dir_with_specified_type(task_dir, 'task')

    model_dir_or_file_path = model_train(
            task_dir_arg,
            data_set,
            lazy_training,
            overwrite,
            max_memory,
            max_processes,
            use_torch
        )
    model_dir_arg = check_dir_with_specified_type(model_dir_or_file_path, 'model', or_file=True)

    _, model_file_names = model_dir_arg
    model_file_name = hyper_parameter_select(model_dir_arg, overwrite, model_file)
    _, task_file_names = task_dir_arg
    model_dir_arg = check_dir_with_specified_type(model_file_name, 'model', or_file=True)
    model_test(
            model_dir_arg,
            data_set,
            n_test,
            overwrite=False,
            max_memory=max_memory,
            max_processes=max_processes,
            use_torch=use_torch
        )
    print('保存模型文件: \'{}\''.format(model_file_name))