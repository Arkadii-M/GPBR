import pickle

def evaluate_task_2D(individual):
    return 1e5
    # global gpbr_eval_2D

    # if gpbr_eval_2D.feasible(individual):
    #     return gpbr_eval_2D.eval(individual)
    # return 1e10

def worker_init_2D(eval_class):
    pass
    global gpbr_eval_2D
    gpbr_eval_2D = eval_class
    # gpbr_eval_2D = pickle.loads(eval_class)


def on_error(err):
    property(err)