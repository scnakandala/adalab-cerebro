import base64
import os
import random
import string
import time
import torch
import dill
import tensorflow as tf


def uuid():
    """
    Utility function to generate unique identifier
    :return:
    """
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))


def mst_identifier(mst):
    """
    Utility function to generate a unique identifier for a MST
    :param mst:
    :return:
    """
    string_id = ""
    keys = [x for x in mst.keys()]
    keys.sort()
    for k in keys:
        v = mst[k]
        string_id = string_id + k + ":" + str(v) + "-"

    string_id = string_id[0:-1]
    return string_id.replace(" ", "_")


def preload_data_helper(data_cache, input_fn_string, input_paths):
    """

    :param data_cache:
    :param input_fn_string:
    :param input_paths:
    :return:
    """
    input_fn = dill.loads(base64.b64decode(input_fn_string))
    for input_path in input_paths:
        if input_path not in data_cache:
            data_cache[input_path] = input_fn(input_path)
    return {"message": "Successfully pre-loaded the data..."}


def _pytorch_load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename+"/model.pth"):
        print("=> loading checkpoint '{}'".format(filename+"/model.pth"))
        checkpoint = torch.load(filename+"/model.pth")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'"
                  .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        
    return model, optimizer


def pytorch_execute_helper(data_cache, epoch, partitions, checkpoint_path, input_paths,
                           input_fn_string, model_fn_string, train_fn_string, mst, train=True):

    begin_time = time.time()

    input_fn = dill.loads(base64.b64decode(input_fn_string))
    model_fn = dill.loads(base64.b64decode(model_fn_string))
    train_fn = dill.loads(base64.b64decode(train_fn_string))

    losses = []
    errors = []
    message = ""
    for partition, input_path in zip(partitions, input_paths):
        if input_path in data_cache:
            # already cached
            data = data_cache[input_path]
        else:
            data = input_fn(input_path) 

        model, opt, loss = model_fn(mst)

        if os.path.exists(checkpoint_path):
            try:
                model, opt = _pytorch_load_checkpoint(model, opt, checkpoint_path)
            except:
                print('Failed to load checkpoint...')

        if torch.cuda.is_available():
            model = model.to(torch.device('cuda'))
            # now individually transfer the optimizer parts...
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(torch.device('cuda'))

                         
        message += "\nEPOCH: %d, PARTITION: %d, Model Building and Session Initialization Time: %d\n" % (
            epoch, partition, time.time() - begin_time)

        time_begin = time.time()
        loss_val, error_val = train_fn(data, model, opt, loss, mst, train=train)
        losses.append(loss_val)
        errors.append(error_val)

        elapsed_time = time.time() - time_begin
        mode = "TRAIN" if train else "VALID"
        message += "EPOCH: %d, PARTITION: %d, %s LOSS: %f, ERROR: %f, Time: %f\n" % (
        epoch, partition, mode, loss_val, error_val, elapsed_time)

        if train:
            begin_time = time.time()
            torch.save({'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}, checkpoint_path+"/model.pth")
            message += "EPOCH: %d, PARTITION: %d, Checkpoint Save Time: %d\n" % (
            epoch, partition, time.time() - begin_time)

    return {'loss': losses, 'error': errors, 'message': message[1:]}


def tf_execute_helper(data_cache, epoch, partitions, checkpoint_path, input_paths,
                      input_fn_string, model_fn_string, train_fn_string, mst, train=True):
    """

    :param data_cache:
    :param epoch:
    :param partitions:
    :param checkpoint_path:
    :param input_paths:
    :param input_fn_string:
    :param model_fn_string:
    :param train_fn_string:
    :param mst:
    :param train:
    :return:
    """
    begin_time = time.time()

    tf.reset_default_graph()

    input_fn = dill.loads(base64.b64decode(input_fn_string))
    model_fn = dill.loads(base64.b64decode(model_fn_string))
    train_fn = dill.loads(base64.b64decode(train_fn_string))

    losses = []
    errors = []
    message = ""
    for partition, input_path in zip(partitions, input_paths):
        tf.reset_default_graph()

        if input_path in data_cache:
            # already cached
            data = data_cache[input_path]
        else:
            data = input_fn(input_path)
            data_cache[input_path] = data

        opt, loss, error = model_fn(data, mst)
        train_step = opt.minimize(loss, colocate_gradients_with_ops=True)
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth = True)
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

        if os.path.exists(checkpoint_path + ".index"):
            saver.restore(sess, checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        message += "\nEPOCH: %d, PARTITION: %d, Model Building and Session Initialization Time: %d\n" % (
            epoch, partition, time.time() - begin_time)

        time_begin = time.time()
        loss_val, error_val = train_fn(sess, train_step, loss, error, train=train)
        losses.append(loss_val)
        errors.append(error_val)

        elapsed_time = time.time() - time_begin
        mode = "TRAIN" if train else "VALID"
        message += "EPOCH: %d, PARTITION: %d, %s LOSS: %f, ERROR: %f, Time: %f\n" % (epoch, partition, mode, loss_val, error_val, elapsed_time)

        if train:
            begin_time = time.time()
            saver.save(sess, checkpoint_path)
            message += "EPOCH: %d, PARTITION: %d, Checkpoint Save Time: %d\n" % (epoch, partition, time.time() - begin_time)

        sess.close()

    return {'loss': losses, 'error': errors, 'message': message[1:]}
