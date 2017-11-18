import os
import tempfile
import unittest

from nnimgproc.util.parameters import Parameters


class TestParametersMethods(unittest.TestCase):

    def test_null(self):
        params = Parameters()
        with self.assertRaises(ValueError):
            params.get('none')
        params.set('none', None)
        self.assertIsNone(params.get('none'))

    def test_default_value(self):
        params = Parameters()
        self.assertEqual([0, 3], params.get('none', [0, 3]))
        self.assertNotEqual([4, 3], params.get('none', [0, 3]))

    def test_set_get(self):
        params = Parameters()
        params.set('none', [0, 3])
        self.assertEqual([0, 3], params.get('none', [4, 3]))
        import numpy as np
        value = np.random.uniform(0, 1, (10, 10))
        params.set('value', value)
        # Set a reasonably small value as threshold.
        self.assertTrue((value - params.get('value')).sum() < 1e-5)

    def test_save_load(self):
        params = Parameters()
        params.set('none', [0, 3])

        import numpy as np
        value = np.random.uniform(0, 1, (10, 10))
        params.set('value', value)
        params.save(os.path.join(tempfile.gettempdir(),
                                 'nnimgproc_test_parameters.pkl'))

        new_params = Parameters()
        new_params.load(os.path.join(tempfile.gettempdir(),
                                     'nnimgproc_test_parameters.pkl'))
        self.assertEqual([0, 3], new_params.get('none'))
        self.assertLess(np.abs((value - new_params.get('value'))).sum(), 1e-5)

        value[5, 5] += 3e-5
        # The value is mutable
        self.assertEqual(value[5, 5], params.get('value')[5, 5])
        # The new value will exceed the neighbourhood of expectation
        self.assertFalse((value[5, 5] - new_params.get('value')[5, 5]) < 1e-5)


if __name__ == '__main__':
    unittest.main()
