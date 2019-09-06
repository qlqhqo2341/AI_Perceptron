import numpy as np
from numpy import random as rand, dot, vectorize
import math
from matplotlib import pyplot as plt
from itertools import product

class act:
    '''
    Activation Function 정의
    원래함수, 미분함수가 담긴 튜플을 반환.
    '''
    linear = (lambda x: x, lambda x: 1)
    relu = (lambda x : x if x>=0 else 0, lambda x : 1 if x>=0 else 0)
    sigmoid = (lambda x: 1/(1+math.exp(-x)),
        lambda x: 1/(1+math.exp(-x))*(1 - 1/(1+math.exp(-x) )) ) 
        
class LayerFile:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        weights = []
        buffer_w = []
        with open(self.filename, "rt") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if len(buffer_w)==0:
                        continue
                    weights.append(buffer_w)
                    buffer_w = []
                else:
                    row = list( map( float, line.split() ) )
                    buffer_w.append(row)

        if len(buffer_w)>0:
            weights.append(buffer_w)
        weights = list( map( np.array, weights) )
        layer = [ *map(len, weights), len(weights[-1][0]) ]
        return layer, weights         

    def save(self, weights):
        with open(self.filename, "wt") as f:
            for w in weights:
                for row in w:
                    print(*row, sep=' ', file=f)
                print(file=f)

class NetData:
    '''
    var = norm_data의 분산
    each_n = norm_data각각의 데이터 수
    '''
    var = 0.01
    each_n = 20
    and_set=(
        np.array([ [0,0],[0,1],[1,0],[1,1],]),
        np.array([ [0],[0],[0],[1],])
    )
    or_set=(
        np.array([ [0,0],[0,1],[1,0],[1,1],]),
        np.array([ [0],[1],[1],[1],])
    )
    xor_set=(
        np.array([ [0,0],[0,1],[1,0],[1,1],]),
        np.array([ [0],[1],[1],[0],])
    )
    donut_set=(
        np.array([ [0,0],[0,0.5],[0,1],[0.5,0],[0.5,0.5],[0.5,1],[1,0],[1,0.5],[1,1], ]),
        np.array([ [0],[0],[0],[0],[1],[0],[0],[0],[0], ])
    )

    @staticmethod
    def _gen_x():
        var, n = NetData.var, NetData.each_n
        x00= np.random.multivariate_normal([0,0], np.eye(2)*var, n)
        x01= np.random.multivariate_normal([0,1], np.eye(2)*var, n)
        x10= np.random.multivariate_normal([1,0], np.eye(2)*var, n)
        x11= np.random.multivariate_normal([1,1], np.eye(2)*var, n)
        x = np.vstack( (x00,x01,x10,x11) )
        return x
    
    @staticmethod
    def _gen_t(correct):
        n = NetData.each_n
        t = np.array( [[1]*n if b==1 else [0]*n for b in correct] )
        return t.reshape(n*4, 1)
    
    @staticmethod
    def and_norm_set():
        x = NetData._gen_x()
        t = NetData._gen_t([0,0,0,1])
        return (x,t)
    
    @staticmethod
    def or_norm_set():
        x = NetData._gen_x()
        t = NetData._gen_t([0,1,1,1])
        return (x,t)
    
    @staticmethod
    def xor_norm_set():
        x = NetData._gen_x()
        t = NetData._gen_t([0,1,1,0])
        return (x,t)

    @staticmethod
    def donut_norm_set():
        var, n = NetData.var, NetData.each_n
        x = []
        for obj in product(range(3), repeat=2):
            lis = np.random.multivariate_normal(np.array(obj)/2, np.eye(2)*var, n)
            x.extend(lis)
        t = [0]*(4*n) + [1]*n + [0]*(4*n)
        return (np.array(x).reshape( (len(x),2) ) , np.array(t).reshape( (len(t),1) ))
            


class NeuralNetwork:
    '''
    뉴럴 네트워크를 구현한 클래스입니다.
    x와 t의 데이터셋에서 한 행은 데이터셋, 한 열은 하나의 노드로 가는 입력값으로 인식합니다.
    '''
    def __init__(self, layerObject, activation, train_x, train_t, last_sigmoid=False):
        '''
        LayerObject가 문자열이면, 파일이름으로 인식하고 파일을 불러옵니다.
            그렇지 않으면 레이어의 리스트로 받고 노드들을 랜덤으로 초기화합니다.
        activation으로 (활성화함수, 미분활성화함수)를 받습니다.
        train_x, train_t로 필요한 데이터를 받습니다.
        classify가 True이면 에러함수를 cross_entropy함수를, False이면 Mean Squared함수를 사용합니다.
        '''
        self.act, self.diff_act = map(np.vectorize, activation)
        if type(layerObject) is str:
            layerFile = LayerFile(layerObject)
            self.layer, self.weights = layerFile.load()
        else:
            self.layer= layerObject
            self.create_weights()

        self.last_act, self.last_diff_act = map(np.vectorize, act.sigmoid) if last_sigmoid else (self.act, self.diff_act)
        
        self.train_x=train_x
        self.train_t=train_t

        if len(self.layer)<2:
            print("Please increases layer size at least 2")
    
    def create_weights(self):
        '''
        뉴럴 네트워크를 만들 때 w를 초기화합니다.
        '''
        weights = self.weights = []
        layer = self.layer
        for i in range(len(layer)):
            if i==len(layer)-1:
                continue
            '''
                [ w  ]
            w = [----]
                [ w0 ]
            '''
            w = np.array( rand.rand( layer[i] + 1, layer[i+1]) )
            weights.append(w)

    def train(self, learning_rate=0.1, repeat=100000, print_num=10, show_error=False, show_line_num=0, mini_batch=False, one_line=False):
        '''
        학습을 시키면서 print_num만큼 중간 과정을 출력합니다.
        show_error로 error를 출력합니다.
        show_line_number에 1이상의 값을 주면,
        인풋2, 아웃풋1일때 결과를 나누는 직선의 변화를 출력합니다.
        값은 출력할 직선의 갯수를 의미합니다.
        '''
        printer = "Step %d : Accuracy = %.8lf, Loss = %.8lf"
        self.learning_rate = learning_rate
        train_x, train_t= self.train_x, self.train_t 
        print_repeat_unit = repeat//print_num
        train_line_unit = repeat//(show_line_num-1) if show_line_num>0 else repeat+1
        error, w_list = [], []
        for i in range(repeat+1):
            # select minibach or one line data set
            if one_line:
                select = np.random.randint(self.train_x.shape[0])
                train_x, train_t = self.train_x[select,:].reshape(1,self.layer[0]), self.train_t[select,:].reshape(1,self.layer[-1])
            elif mini_batch:
                select = []
                while len(select)==0:
                    select = np.random.randint(2,size=self.train_x.shape[0])
                    select = [i for i,v in enumerate(select) if v==1]
                train_x, train_t = self.train_x[select,:], self.train_t[select,:]
            else:
                train_x, train_t = self.train_x, self.train_t
                
            # 현재 훈련데이터로 학습후 업데이트
            output, flow_net, flow_out = self.predict(train_x)
            self.back_propagation(self.get_diff_loss(output,train_t),
                flow_net, flow_out, train_x)

            # 에러리스트 추가
            out = self.predict()[0]
            acc_loss = self.get_accuracy_loss(out, self.train_t)
            error.append((i,acc_loss[1]))
            if i%print_repeat_unit==0:
                print(printer%(i, *acc_loss ) )

            if i%train_line_unit==0 and show_line_num>0:
                w_list.append(
                    (i,
                    np.array(self.weights[-1],copy=True))
                )
            
        if (show_error==True):
            self.show_error_plot(error)
        if show_line_num>0:
            self.show_final_plot(w_list)
                
    def test(self, test_x=None, test_t=None, show_w_plot=False):
        '''
        test_x과 test_t의 정확도와 loss를 계산합니다.
        show_w_plot=True로 하면
        show_w_plot의 함수기능도 수행합니다.
        '''
        printer = "Test result : Accuracy = %.8lf, Loss = %.8f"
        test_x, test_t = test_x if not test_x is None else self.train_x , test_t if not test_t is None else self.train_t 

        output, flow_net, flow_output = self.predict(test_x)
        print(printer%(self.get_accuracy_loss(output,test_t)))
        if show_w_plot:
            self.show_w_plot(flow_output,test_x,test_t)

    def show_error_plot(self, error_list, first_cut_off=True):
        '''
        error_list를 각각 (i,error)로 받아서
        그래프로 표시해줍니다.
        first_cut_off로 초반의 에러 표시를 버릴수 있습니다.
        '''
        fig = plt.figure(figsize=(5,5))
        plot = fig.add_subplot(1,1,1)

        #Error line
        error_list = error_list[len(error_list)//10:] if first_cut_off else error_list
        # 번호와 에러값을 분리해서 각각의 리스트로 담음.
        i_list, error_list= zip(*error_list)
        plot.plot(i_list, error_list, label="error in iteration")

        #y=0 line
        x,y=[0,max(i_list)],[0,0]
        plot.plot(x,y,label="error=0 line.",linestyle="dotted")
        plot.legend()

        plt.show()              

    def show_w_plot(self, flow_out, train_x, train_t):
        '''
        네트워크의 노드들 중에서 Input이 2인 노드들을
        각각 직선으로 표시합니다.
        '''
        show_list = [ (i+1, flow_out[i-1] if i>0 else train_x , w) for i,w in enumerate(self.weights) if w.shape[0]==3]
        length = len(show_list)
        height = max(map(lambda x : x[2].shape[1],show_list))

        plots_list = []
        fig = plt.figure(figsize=(5,5))
        for i,(n,x,w) in enumerate(show_list):
            nodes = []
            
            t,f=[[],[],], [[],[]]
            for _, target in enumerate(train_t):
                lis = t if target==1 else f
                lis[0].append(x[_][0])
                lis[1].append(x[_][1])

            for j in range(w.shape[1]):
                plot = fig.add_subplot(height, length, i+1+(length*j))
                nodes.append( plot )
                #line
                linex = np.linspace(np.min(x[:,0])-0.5, np.max(x[:,0])+0.5, 100)
                liney = -(w[0][j]*linex + w[2][j] - 0.5)/(w[1][j])
                plot.plot(linex,liney,label="divide")
                #dot
                plot.scatter(t[0],t[1],marker='o')
                plot.scatter(f[0],f[1],marker='x')
                               
            plots_list.append(nodes)
        plt.show()
        

    def show_final_plot(self, w_list=None):
        '''
        최종 결과값에 대한 그래프를 표시.
        w_list가 있으면, 각각의 (i,w)에 대한 직선들을 표시합니다.
        '''
        train_x, train_t = self.train_x, self.train_t
        x = train_x
        w = self.weights[-1]

        fig = plt.figure(figsize=(5,5))
        subplot = fig.add_subplot(1,1,1)
        subplot.set_xlim([-0.5,1.5])
        subplot.set_ylim([-0.5,1.5])
        
        #train dot
        t,f=[[],[],], [[],[]]
        for i, target in enumerate(train_t):
            lis = t if target==1 else f
            lis[0].append(train_x[i][0])
            lis[1].append(train_x[i][1])
        subplot.scatter(t[0],t[1],marker='o')
        subplot.scatter(f[0],f[1],marker='x')

        # x,y범위 설정
        x_range = ( min(train_x[:,0]), max(train_x[:,0]) )
        y_range = ( min(train_x[:,1]), max(train_x[:,1]) )
        mid_x = (x_range[1] - x_range[0])/2
        mid_y = (y_range[1] - y_range[0])/2
        x_range = (x_range[0] - mid_x, x_range[1] + mid_x)
        y_range = (y_range[0] - mid_y, y_range[1] + mid_y)
        precision = 100

        # lines draw
        if not w_list is None:
            for i,w in w_list:
                linex = np.linspace(*x_range, 100)
                # w0+w1x1+w2x2 = 0.5
                # x2 = -(w0+w1x1-0.5)/w2
                liney = -(w[2][0] + w[0][0]*linex - .5)/w[1][0]
                subplot.plot(linex, liney, label="iter=%d, divide_line"%(i))
            subplot.legend()


        # result draw
        space = [ (x1,x2) for x2 in np.linspace(*y_range,precision) \
            for x1 in np.linspace(*x_range,precision)]
        
        output = self.predict(np.array(space))[0]
        results = output.reshape((precision,precision))        
        subplot.imshow(results, origin='lower', extent=(*x_range,*y_range),
            cmap=plt.cm.gray_r, alpha=0.5)
        
        plt.show()

    def predict(self, train_data=None):
        '''
        train_data로 예측합니다.
        최종 output, 중간net, 중간output리스트를 반환합니다.
        '''
        train_data = train_data if not train_data is None else self.train_x
        flow_net, flow_output = [], []
        data = np.array(train_data, copy=True)
        act_func = self.act
        last_act_func = self.last_act
        for i, w in enumerate(self.weights):
            '''
            data_added_1 = [ data | 1(x0) ]
            '''
            data_added_1 = self.add_x0(data)
            # w와 x를 곱해서 net을 만듬.
            net = dot(data_added_1,w)
            flow_net.append(net)
            output = act_func(net) if i<len(self.weights)-1 else last_act_func(net) 
            flow_output.append(output)
            data = np.array(output, copy=True)
        return data, flow_net, flow_output
    
    def get_diff_loss(self, output, target):
        '''
        최종 에러에 대한 미분을 반환합니다.
        '''
        return -(target-output)

    def get_accuracy_loss(self, output, target):
        '''
        정확도와 loss를 반환합니다.
        '''
        correct = ((output-0.5)*(target-0.5)).flatten()
        correct = list(map(lambda x : True if x>0 else False, correct))
        return correct.count(True)/len(correct), sum(np.square(output-target))        

    def back_propagation(self, diff_loss, flow_net, flow_out, train_x):
        '''
        주어진 loss의 미분값과 중간결과값들로
        해당 network의 weights를 업데이트시킵니다.
        '''
        diff_func = self.diff_act
        act_func = self.act
        weights = self.weights
        
        # shape : train_x, out_nodes.
        delta = -diff_loss*self.last_diff_act(flow_net[-1])
        for i in range(len(flow_net)-1,-1,-1):
            w = weights[i]
            net = flow_net[i]
            out = flow_out[i]
            prev_out = flow_out[i-1] if i>0 else train_x
            prev_out_w0 = self.add_x0(prev_out)
            
            # new delta = diff_func(prev_net) * sum( delta * w )
            # delta(train_x, output_nodes) * w(prev_output_nodes + 1, output_nodes)
            # delta(train_x, output_nodes) * w.transpose(output_nodes, prev_output_nodes+1) 
            if(i>0):
                new_delta = diff_func(flow_net[i-1]) * dot(delta, w.transpose())[:,:-1]
            else:
                new_delta = delta

            # delta w = c * delta * x
            # delta(train_x, output_nodes) * prev_out(train_x, prev_output_nodes+1)
            # prev_out.transpose(prev_output_nodes+1, train_x) * delta(train_x, output_nodes)
            # dw = prev_output_nodes+1, output_nodes
            dw = self.learning_rate * dot(prev_out_w0.transpose(), delta)
            
            # 각각 train_set에 대한 평균으로 적용.
            dw /= w.shape[0]

            weights[i] = w  + dw
            delta = new_delta
            
    def save(self, filename):
        '''
        현재 네트워크의 weights 정보를 저장합니다.
        '''
        layerFile = LayerFile(filename)
        layerFile.save(self.weights)

    def add_x0(self, array, axis=1):
        '''
        해당 행렬에 오른쪽 열 (axis=1,default) 혹은 아래쪽 행 (axis=0)에 1값을 추가합니다.
        '''
        if len(array.shape)==1 and axis==1:
            return np.hstack( ( array, np.ones(1) )  )

        other_axis=1-axis
        s = array.shape[other_axis]
        if axis==0:
            w0 = np.ones( ( 1, s ) )
            return np.vstack( ( array, w0 ) ) 
        else:
            w0 = np.ones( ( s, 1 ) )
            return np.hstack( ( array, w0 ) )