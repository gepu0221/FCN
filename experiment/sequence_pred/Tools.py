import numpy as np

#Queue
class Queue():

    def __init__(self, size):
        self.size = size
        self.front = -1
        self.rear = -1
        self.queue = []

    #enter the queue
    def enqueue(self, elem):
        if self.isfull():
            raise Exception('queue is full')
        else:
            self.queue.append(elem)
            self.rear = self.rear+1

    #depart the queue
    def dequeue(self):
        if self.isempty():
            raise Exception('queue is empty')
        else:
            self.queue.pop(0)
            self.front = self.front+1


    def isfull(self):
        return self.rear-self.front == self.size

    def isempty(self):
        return self.rear == self.front

    def showQueue(self):
        print(self.queue)
    
    #if get first %num data of queue
    def showFirstNum(self, first_num):
        if len(self.queue) < first_num:
            raise Exception('queue dose not have so much data')
        else:
            return True
'''                
q=Queue(3)
for i in range(3):
    q.enqueue(i)
    print('front: %d, rear: %d' % (q.front, q.rear))
    q.showQueue()

for i in range(2):
    q.dequeue()
    print('front: %d, rear: %d' % (q.front, q.rear))
    q.showQueue()

q.enqueue(10)
print('front: %d, rear: %d' % (q.front, q.rear))
q.showQueue()
q.enqueue(20)
print('front: %d, rear: %d' % (q.front, q.rear))
q.showQueue()

for i in range(3):
    q.dequeue()
    print('front: %d, rear: %d' % (q.front, q.rear))
    q.showQueue()
'''
