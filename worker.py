import base64
import sys
import threading
import gc
import traceback
from xmlrpc.server import SimpleXMLRPCServer

import dill

server = SimpleXMLRPCServer((sys.argv[1], int(sys.argv[2])), allow_none=True)

data_cache = {}
status_dict = {}


def initialize_worker():
    """
    Initialize the worker by resetting the caches
    :return:
    """
    global data_cache
    global status_dict
    del data_cache
    del status_dict
    gc.collect()
    data_cache = {}
    status_dict = {}


def execute(exec_id, code_string, params):
    # can execute only one at a time
    """
    :param exec_id:
    :param code_string:
    :param params:
    :return:
    """
    if len([y for y in status_dict.values() if y["status"] == "RUNNING"]) > 0:
        return base64.b64encode(dill.dumps("BUSY"))
    else:
        func = dill.loads(base64.b64decode(code_string))

        def bg_execute(exec_id, func, params):
            """
            :param exec_id:
            :param func:
            :param params:
            """
            try:
                func_result = func(data_cache, *params)
                status_dict[exec_id] = {"status": "COMPLETED", "result": func_result}
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                sys.stdout.flush()
                status_dict[exec_id] = {"status": "FAILED"}

        status_dict[exec_id] = {"status": "RUNNING"}
        thread = threading.Thread(target=bg_execute, args=(exec_id, func, params,))
        thread.start()

        return base64.b64encode(dill.dumps("LAUNCHED"))


def status(exec_id):
    """
    :param exec_id:
    :return:
    """
    if exec_id in status_dict:
        return base64.b64encode(dill.dumps(status_dict[exec_id]))
    else:
        return base64.b64encode(dill.dumps({"status": "INVALID ID"}))


def is_live():
    return True


server.register_function(execute)
server.register_function(status)
server.register_function(initialize_worker)
server.register_function(is_live)
server.serve_forever()
