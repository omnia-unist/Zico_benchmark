import tensorflow as tf

# CPU_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable', 'Assign', 'Identity', 'Relu']
# CPU_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable', 'Assign', 'Identity']
# CPU_OPS = ['Relu'] 
CPU_OPS = ['Identity']

def assign_to_device(gpu, cpu='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in CPU_OPS:
            return "/" + cpu
        else:
            return gpu

    return _assign


def assign_to_gpu(gpu):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in CPU_OPS:
            return gpu
        else:
            return gpu

    return _assign
