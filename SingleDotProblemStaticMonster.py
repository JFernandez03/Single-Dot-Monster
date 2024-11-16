import random

class Monster:
    def __init__ (self, color, pos):
        self.pos = pos
        self.color = color
    def display(self):
        print(self.color, self.pos)
    def __eq__(self, m):
        return m.pos == self.pos and self.color == m.color
    def __hash__(self):
        return hash((self.pos, color))
    def __copy__(self):
        return Monster(self.color, self.pos)
    def __deepcopy__(self):
        return Monster(self.color, self.pos)

class Agent:
    def __init__(self, pos):
        self.pos = pos
    def display(self):
        print(self.pos)
    def __eq__(self, a):
        return a.pos == self.pos
    def __hash__(self):
        return hash(self.pos)
    
class State:
    def __init__(self, agentPos):
        self.agentClass = Agent(agentPos)

    def display(self):
        print('State: ')
        print('agent position: ', self.agentClass.pos)
        
    def __eq__(self, state):
        if state.agentClass.pos != self.agentClass.pos: return False
        return True

    def __hash__(self):
        return hash(self.agentClass.pos)

# check:
#s = State((1, 1), [Monster('R', (1,1))])
#q = State(s.agentClass.pos, s.monsterClassList.deepcopy())
#q.agentClass.pos = (2, 2)
#q.monsterClassList[0].pos = (3, 3)
#q.display()
#s.display()

class Problem:
    def __init__(self, mazeFile):
        self.__wall = '*'
        self.__agent = 'P'
        self.__monsters = ['R', 'G', 'B']
        self.__dot = '.'
        self.__statesExplored = 0
        self.potential_moves = {'R': (1, 0), 'L': (-1, 0), 'D': (0, 1), 'U': (0, -1)}

        self.readMaze(mazeFile)

    def readMaze(self, mazeFile):
        self.walls = []
        self.pacman = 0
        self.monsters = []
        self.dots = []
        self.xMax = 0
        self.yMax = 0
        
        f = open (mazeFile, 'r')
        if not f:
            print('File not found')
            return False        
        y = 0
        while True:
            s = list (f.readline ())
            #print (s)
            
            if s == []: break
            if s[-1] == '\n': s.pop()
            x=0
            for k in s:
                if k == self.__wall: self.walls.append ((x, y))
                if k == self.__agent: self.pacman = (x, y)
                if k == self.__dot: self.dots.append((x, y))
                if k in self.__monsters:
                    self.monsters.append(Monster(k, (x, y)))
                x += 1
            y += 1
            self.xMax = max (self.xMax, x)
        self.yMax = y
        return True

    def isWall(self, pos):
        return pos in self.walls

    def getStartState(self):
        pacman = (self.pacman[0], self.pacman[1])
        return State(pacman)

    def isValidMove(self, pos):
        x, y = pos
        return 0 <= x < self.xMax and 0 <= y < self.yMax and not self.isWall(pos)

    def isTerminal(self, state):
        if state.agentClass.pos in self.dots: return True
        for m in self.monsters:
            if m.pos == state.agentClass.pos: return True
        return False

    def reward(self, state):
        if state.agentClass.pos in self.dots: return 10
        else:
            for m in self.monsters:
                if m.pos == state.agentClass.pos: return -10
        return 0

    def transition(self, state):
        x, y = state.agentClass.pos

        potential_moves = [((x+1, y), 'R'), ((x-1, y), 'L'),
                           ((x, y+1), 'D'), ((x, y-1), 'U')]

        moves = [(State(move), a) for move, a in potential_moves if self.isValidMove(move)]
        self.__statesExplored += 1
        return moves

    def moveAgent(self, agent, a):
        x, y = agent
        if a == 'L': x -= 1
        if a == 'R': x += 1
        if a == 'U': y -= 1
        if a == 'D': y += 1
        
        return (x, y)
            
