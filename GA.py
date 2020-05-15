import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

input_data=np.array([[1,1,1],[3,3,4],[4,4,5]])
t=np.ones(3)#输入数据和输出数据

n,m,l=3,5,1#输入神经元，隐层神经元，输出神经元个数

def initial(n,m,l,N):#依据指数分布生成随机数组
    w=np.zeros((N,n*m+m*l),np.float)
    for i in range(N):
            r=np.random.rand((n*m+m*l))
            w[i]=-np.log(r)
    return w

def sigmoid(data):
    return 1/(1+np.exp(-data))

def cost(choice_w):
    n,m,l=3,5,1#计算某一参数对应的代价函数大小,三层神经网络,计算一组参数对应的一个适应度值
    w_hide=np.array(choice_w[:n*m].reshape(n,m))
    w_out=np.array(choice_w[n*m:]).reshape(m,l)
    fitness=0
    n=0
    for data in input_data:   
        hide_input=np.dot(data,w_hide)
        hide_output=sigmoid(hide_input)
        out_in=np.dot(hide_output,w_out)
        y=sigmoid(out_in)#前向传播
        fitness+=np.abs(y-t[n])
        n+=1
    return 1/fitness[0]

def explore(now_w,max_iter,n,Gbest_w,Ibest_w):
    w2=0.3
    w1=0.4
    p1=w2+(max_iter-n)/max_iter*w1
    p2=p1+(1-p1)/2
    r=np.random.rand()
    b=np.random.rand()
    start=np.random.randint(0,len(now_w)-1)
    end=np.random.randint(start,len(now_w))
    new=now_w
    if r<=p1:
        ano=np.random.randint(0,len(w))
        ano_w=w[ano]
        new[start:end]=now_w[start:end]*(1-b)+ano_w[start:end]*b
    elif r>=p1 and r<=p2:
        new[start:end]=now_w[start:end]*(1-b)+Gbest_w[start:end]*b
    else:
        new[start:end]=now_w[start:end]*(1-b)+Ibest_w[start:end]*b
    #print(new)
    #print('OK')
    return new
    

def explict(now_w,p):
    if p>0.7:
        for k in range(2):
            r=np.random.rand()
            pos=np.random.randint(0,len(now_w))
            now_w[pos]=-np.log(r)
    if p<0.7:
     r=np.random.rand()
     pos=np.random.randint(0,len(now_w))
     now_w[pos]=-np.log(r)
     #print('YES')
    return now_w


def create_new(w):#根据排序选择新的数组
    fit=np.zeros(len(w))
    for i in range(len(w)):
        fit[i]=cost(w[i])
    frame=DataFrame(w)
    frame['fit']=fit
    frame=frame.sort_values(by='fit')
    frame=frame.drop('fit',axis=1)
    w=frame.values
    len_one=6*len(w)//10
    len_two=8*len(w)//10
    len_three=len(w)
    new=np.zeros(len(w))
    for i in range(len_one):
        c=np.random.randint(0,len(w)//3)
        new[c]=1
    for i in range(len_one,len_two):
        c=np.random.randint(len(w)//3,2*len(w)//3)
        new[c]=1
    for i in range(len_two,len_three):
        c=np.random.randint(2*len(w)//3,len(w))
        new[c]=1
        
    frame['drop']=new
    frame=frame[frame['drop']!=0]
    frame=frame.drop('drop',axis=1)
    #print(frame)
    return frame.values
    
    
    
    
    
def GA(w,max_iter,max_choice,n,m,l):#max_iter最大迭代次数m
    Gbest=cost(w[0])
    Ibest=Gbest
    Ibest_w,Gbest_w=w[0],w[0]
    number=0
    while(1):###？？？是否有必要三层循环？？？
        number+=1#n记录迭代次数
        for i in range(max_choice):
            choice=np.random.randint(0,len(w))
            for j in range(3):
                r=np.random.rand()##确定随机概率 r=0.3
                if r>0.3:
                    fitness=cost(w[choice])
                    distance_g=np.abs(fitness-Gbest)
                    distance_I=np.abs(fitness-Ibest)
                    p=distance_g/(distance_I+distance_g)
                    p=max(p,0.1)
                    p=min(p,0.9)###P表示对于抽出来的参数选择交叉的概率
                else:
                    p=np.random.rand()
                    
                r=np.random.rand()
                if r>=p:
                    new=explore(w[choice],max_iter,n,Gbest_w,Ibest_w)
                else:
                    new=explict(w[choice],1-r)###是否需要去除旧的参数？？？
                    
                #print(new)
                np.insert(w,len(w),new,axis=0)
                
                if cost(new)<Ibest:
                    Ibest=cost(new)
                    Ibest_w=new
                    
        w=create_new(w)
        
        if Ibest<Gbest:
            Gbest=Ibest
            Gbest_w=Ibest_w
        if Gbest<1 or number>=max_iter:
            print(number)
            return Gbest,Gbest_w


            
w=initial(n,m,l,10) 
best,result=GA(w,100,2,n,m,l)         
        
        