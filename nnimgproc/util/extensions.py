import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary


# Simple tensorboard interface which only supports scalar values
class TensorboardWriter(object):
    def __init__(self, output: str):
        """

        :param output: string, folder containing the log file
        """
        self._writer = tf.summary.FileWriter(logdir=output, graph=None,
                                             flush_secs=30)

    def add_entry(self, tag: str, value: float, time: int):
        """
        Put one new value to the tensorboard log

        :param tag: string, name of the value
        :param value: float, value
        :param time: int, index of the value, usually step or epoch
        :return:
        """
        summary_value = Summary.Value(tag=tag, simple_value=value)
        entry = Summary(value=[summary_value])
        self._writer.add_summary(entry, time)

    def close(self):
        if self._writer is not None:
            self._writer.close()
