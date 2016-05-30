# This is a variable scope aware configuation object for TensorFlow

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class Config:
    def __init__(self):
        root = self.Scope('')
        for k, v in FLAGS.__dict__['__flags'].iteritems():
            root[k] = v
        self.stack = [ root ]

    def _pop_stale(self):
        var_scope_name = tf.get_variable_scope().name
        top = self.stack[0]
        while not top.contains(var_scope_name):
            # We aren't in this scope anymore
            self.stack.pop(0)
            top = self.stack[0]

    def __getitem__(self, name):
        self._pop_stale()            
        # Recursively extract value
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return cs[name]

        raise KeyError(name)

    def __setitem__(self, name, value):
        self._pop_stale()            
        top = self.stack[0]
        var_scope_name = tf.get_variable_scope().name
        assert top.contains(var_scope_name)

        if top.name != var_scope_name:
            top = self.Scope(var_scope_name)
            self.stack.insert(0, top)

        top[name] = value

    class Scope(dict):
        def __init__(self, name):
            self.name = name

        def contains(self, var_scope_name):
            return var_scope_name.startswith(self.name)



# Test
if __name__ == '__main__':

    def assert_raises(exception, fn):
        try:
            fn()
        except exception:
            pass
        else:
            assert False, "Expected exception"

    c = Config()

    c['hello'] = 1
    assert c['hello'] == 1

    with tf.variable_scope('foo'):
        c['bar'] = 2
        assert c['bar'] == 2
        assert c['hello'] == 1

        with tf.variable_scope('meow'):
            c['dog'] = 3
            assert c['dog'] == 3
            assert c['bar'] == 2
            assert c['hello'] == 1

        assert_raises(KeyError, lambda: c['dog'])
        assert c['bar'] == 2
        assert c['hello'] == 1


