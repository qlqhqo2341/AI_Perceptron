def and_test():
        from network import act, NeuralNetwork, NetData

        net = NeuralNetwork([2,1], act.sigmoid, *NetData.and_set)
        net.train(repeat=2000, show_error=True, show_line_num=4)
        net.save("net/and.txt")

def or_test():
        from network import act, NeuralNetwork, NetData

        net = NeuralNetwork([2,1], act.sigmoid, *NetData.or_set)
        net.train(repeat=2000, show_error=True, show_line_num=4)
        net.save("net/or.txt")

def xor_test():
        from network import act, NeuralNetwork, NetData

        net = NeuralNetwork([2,2,1], act.sigmoid, *NetData.xor_set)
        net.train(repeat=30000, show_error=True)
        net.test(show_w_plot=True)
        net.show_final_plot()
        net.save("net/xor.txt")

def donut_test():
        from network import act, NeuralNetwork, NetData

        net = NeuralNetwork([2,10,1], act.sigmoid, *NetData.donut_set)
        net.train(learning_rate=1.0, repeat=50000, show_error=True)
        net.show_final_plot()
        net.save("net/donut.txt")

def donut_low_learn_test():
        from network import act, NeuralNetwork, NetData

        net = NeuralNetwork([2,10,1], act.sigmoid, *NetData.donut_set)
        net.train(learning_rate=0.1, repeat=50000, show_error=True)
        net.show_final_plot()
        net.save("net/donut_low_learn.txt")

def donut_norm_test():
        from network import act, NeuralNetwork, NetData

        net = NeuralNetwork([2,10,1], act.sigmoid, *NetData.donut_norm_set())
        net.train(learning_rate=0.1, repeat=50000, show_error=True)
        net.show_final_plot()
        net.save("net/donut_norm.txt")

def compare_relu_sigmoid_test():
        from network import act, NeuralNetwork, NetData

        def func(actor, mini_batch):
        for i in range(5):
                net = NeuralNetwork([2,6,1], actor, *NetData.and_norm_set(), last_sigmoid=True)
                net.train(learning_rate=0.1, repeat=200, print_num=1, mini_batch=mini_batch)
                print()

        print("relu-minibatch")
        func(act.relu, True)

        print("sigmoid-non-minibatch")
        func(act.sigmoid, False)

def relu_test():
        from network import act, NeuralNetwork, NetData

        def func(actor, mini_batch):
                for i in range(10):
                        net = NeuralNetwork([2,2,1], actor, *NetData.xor_norm_set(), last_sigmoid=True)
                        net.train(learning_rate=0.1, repeat=1500, print_num=1, mini_batch=mini_batch)
                        print()
        print("relu-minibatch")
        func(act.relu, True)
