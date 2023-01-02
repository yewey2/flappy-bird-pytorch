import random, math, time, copy
 
# GLOBALS
VELOCITY_X = 6
 
# PYGAME
RENDERING = True
RENDERING = False

SCREENWIDTH  = 600 # pixels
SCREENHEIGHT = 600 # pixels
CAPTION = "Flappy Bird"
 
if RENDERING:
    import pygame
    pygame.init()
    window = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption(CAPTION)
    CLOCK = pygame.time.Clock()
    FRAMERATE = 40
    
    DEFAULT_FONT = pygame.font.SysFont("arial",30)
    
    START_TEXT = DEFAULT_FONT.render("PRESS SPACE TO START", True, (0,0,0))
else:
    window = None
 
 
class Rect:
    def __init__(self, x,y,w,h):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
 
        if RENDERING:
            self.color = (random.randint(0,255),
                          random.randint(0,255),
                          random.randint(0,255))
    def move(self, x=0, y=0):
        self.x+=x
        self.y+=y
 
    def colliderect(self, rect):
        left  = max(self.x, rect.x)
        top   = max(self.y, rect.y)
        right = min(self.x + self.w, rect.x + rect.w)
        bot   = min(self.y + self.h, rect.y + rect.h)
        if right-left > 0 and bot-top > 0:
            return ((left-right) * (top-bot)) > 0
        return False
 
    def collidepoint(self, first=None, second=None):
        if second:
            x=first
            y=second
        else:
            try:
                x,y = first
            except Exception as e:
                print(e)
        return self.x <= x <= self.x+self.w and self.y <= y <= self.y+self.h
        
    def get_position(self):
        return (int(self.x), int(self.y), self.w, self.h)
 
    def draw(self, win):
        if not RENDERING:
            return
        pygame.draw.rect(win, self.color, self.get_position())
 
class Game_Object(Rect):
    pass
 
class Bird(Game_Object):
    BIRD_X = 300
    BIRD_Y = 100
    BIRD_WIDTH = 15
    BIRD_HEIGHT = 15
    GRAVITY = 1.2
    JUMP_HEIGHT = 12
    BIRD_MAX_VEL = 20
    
    def __init__(self):
        self.x = self.BIRD_X
        self.y = self.BIRD_Y
        self.w = self.BIRD_WIDTH
        self.h = self.BIRD_HEIGHT
        self.vel = 0
        self.max_vel = self.BIRD_MAX_VEL
        self.jump_height = self.JUMP_HEIGHT
        self.gravity = self.GRAVITY
        self.action = 0
        self.alive=True
 
        if RENDERING:
            self.color = (random.randint(0,255),
                          random.randint(0,255),
                          random.randint(0,255))
            
    def move(self, jumping = False):
        if self.alive:
            if jumping:
                if self.vel < -self.jump_height/3: #double jump
                    self.vel = -self.jump_height * 1.3
                else: #single jump
                        self.vel = -self.jump_height
            self.y += self.vel
            if self.y<=0: 
                self.vel=0
            self.y = max(self.y, 0)
            self.y = min(SCREENWIDTH-self.h, self.y)
            self.vel += self.gravity
            self.vel = min(self.vel,self.max_vel)
        else:
            self.x -= VELOCITY_X
            self.y += self.vel
            self.y = max(self.y, 0)
            self.y = min(SCREENWIDTH-self.h, self.y)
            self.vel += self.gravity
            self.vel = min(self.vel,self.max_vel)
 
    def kill(self):
        if self.alive:
            # self.vel = -self.jump_height # Messes up the machine learning model!!!
            self.alive=False
 
class Wall(Game_Object):
    def __init__(self, x,y,w,h):
        super().__init__(x,y,w,h)
        
    def collide(self, bird):
        left  = max(self.x, bird.x);
        top   = max(self.y, bird.y);
        right = min(self.x + self.w, bird.x + bird.w);
        bot   = min(self.y + self.h, bird.y + bird.h);
 
        if right-left > 0 and bot-top > 0:
            return ((left-right) * (top-bot)) > 0
 
        return False
 
 
    def move(self, vel):
        self.x-=vel
        
class Scorebox(Game_Object):
    def __init__(self, x,y,w,h):
        super().__init__(x,y,w,h)
        self.score = False
 
    def move(self, vel):
        self.x-=vel
 
    def collide(self, bird):
        if not self.score:
            left  = max(self.x, bird.x);
            top   = max(self.y, bird.y);
            right = min(self.x + self.w, bird.x + bird.w);
            bot   = min(self.y + self.h, bird.y + bird.h);
 
            if right-left > 0 and bot-top > 0:
                self.score = ((left-right) * (top-bot)) > 0
                return self.score
            return False
        return False
        
 
 
class Column: #group 2 walls together
    WIDTH=80
    OPENING_SIZE = 500
    def __init__(self, x, w, opening_size):
        self.x = x
        self.w = w
        self.opening_size = opening_size
        self.set_opening()
 
    def set_opening(self):
        self.opening_top = random.randint(50,SCREENHEIGHT-50-self.opening_size)
        self.top_wall = Wall(self.x, 0, self.w, self.opening_top)
        self.bot_wall = Wall(self.x, self.opening_top+self.opening_size, self.w, SCREENHEIGHT-self.opening_top-self.opening_size)
        self.opening = Scorebox(self.x+self.w, self.opening_top, 1, self.opening_size)
 
    # Override
    def get_position(self):
        return (self.opening.x, self.opening.y + self.opening.h//2)
 
    def move(self, vel=VELOCITY_X):
        self.x -= vel
        self.top_wall.move(vel)
        self.bot_wall.move(vel)
        self.opening.move(vel)
        if self.x < -self.w:
            self.x += SCREENWIDTH+self.w
            self.set_opening()
            
    def collide(self,bird):
        top = self.top_wall.collide(bird)
        bot = self.bot_wall.collide(bird)
        return top or bot
 
    def score(self,bird):
        return self.opening.collide(bird)
 
 
    def draw(self,win):
        self.top_wall.draw(win)
        self.bot_wall.draw(win)

if __name__ == '__main__':
    pygame.quit()
