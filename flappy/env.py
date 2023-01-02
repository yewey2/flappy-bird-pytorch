from setup import *

SHUTDOWN = None

class Env():
    def __init__(self):
        self.bird=Bird()
        self.column1 = Column(SCREENWIDTH,Column.WIDTH,Column.OPENING_SIZE)
        self.column2 = Column(SCREENWIDTH*3//2 + Column.WIDTH//2, Column.WIDTH,Column.OPENING_SIZE)
        self.reward = 0
        self.count=0
        self.done=False
 
    def reset(self):
        self.__init__()
        return self.get_state()
 
    def get_state(self):
        state_str = ''
        state = tuple()
        state += (self.bird.get_position()[1] / SCREENHEIGHT,)
        state_str += f'bird pos: {state[0]:.02f}, '
        state += (self.bird.vel / Bird.BIRD_MAX_VEL,)
        state_str += f'bird vel: {state[1]:.02f}, '
 
        pos1 = self.column1.get_position()[:2]
        pos1 = (pos1[0] / SCREENWIDTH, pos1[1] / SCREENHEIGHT)
        
        pos2 = self.column2.get_position()[:2]
        pos2 = (pos2[0] / SCREENWIDTH, pos2[1] / SCREENHEIGHT)

        if pos1[0] < pos2[0]:
            state += pos1+pos2
        else:
            state += pos2+pos1
            
        state_str += f'col 1: {state[2]:.02f} {state[3]:.02f}, '
        state_str += f'col 1: {state[4]:.02f} {state[5]:.02f}'
        state = list(state)
        # print([f'{i:.3}' for i in state])
        # print(state_str)
        return state
 
        
    def bird_movement(self):
        self.reward = 0
        if self.bird.action == 1: # jump
            self.bird.move(True)
            # self.reward -= 0.03 # Punished for jumping to decrease jumping rate
        else: # not jumping
            self.bird.move()
 
        if self.column1.score(self.bird) or self.column2.score(self.bird):
            self.reward += 3
            self.count += 1
 
        if self.column1.collide(self.bird) or self.column2.collide(self.bird):
            self.bird.kill()
 
        if self.bird.y+self.bird.h > SCREENWIDTH:
            self.bird.kill()
 
        if not self.bird.alive: #bird is dead, give negative reward
            self.reward = -10
        else: #bird is alive
            self.reward += 0.01 # rewarded for being alive
 
    def runframe(self, action1):
        self.done = False
        self.bird.action = action1
        self.column1.move()
        self.column2.move()
        self.bird_movement()
        state = self.get_state()
 
        if not self.bird.alive:
            self.done=True
        
        if self.count>20:
            self.done=True
 
        return state, self.reward, self.done
 
    def render(self):
        global SHUTDOWN
        CLOCK.tick(FRAMERATE*1.5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                SHUTDOWN = True
                print('Closing pygame')
        window.fill((255,255,255))
        for item in [self.bird, self.column1, self.column2]:
            item.draw(window)
        score = self.count
        score_text = DEFAULT_FONT.render(f'{score}', True, (0,0,0))
        r = score_text.get_rect(center=(SCREENWIDTH//2,50))
        window.blit(score_text, r)
        pygame.display.update()
        return SHUTDOWN
 
 
 
 