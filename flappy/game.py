from setup import *

FRAMERATE = 60

# Column.WIDTH = 100
Column.OPENING_SIZE = 180
# Bird.BIRD_X = SCREENWIDTH//3
# Bird.BIRD_Y = SCREENWIDTH//2 - 100
# Bird.BIRD_WIDTH= 20 
# Bird.BIRD_HEIGHT = 20
#Bird.BIRD_MAX_VEL = 15
#Bird.JUMP_HEIGHT = 13

def render_game(win, game_objects,score):
    win.fill((255,255,255))
    for item in game_objects:
        item.draw(win)
    score_text = DEFAULT_FONT.render(f'{score}', True, (0,0,0))
    r = score_text.get_rect(center=(SCREENWIDTH//2,50))
    win.blit(score_text, r)
    pygame.display.update()

def runframe(win, 
        bird = None,
        column1 = Column(SCREENWIDTH,100,150), 
        column2 = Column(SCREENWIDTH,100,150),
        score = list(),
        ):
    if bird is None:
        bird = Bird()
    CLOCK.tick(FRAMERATE)
    end = False
    jumping = False
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            end = 'shutdown'
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                jumping = True
                
    if column1.score(bird) or column2.score(bird):
        score.append(1)

    if column1.collide(bird) or column2.collide(bird):
        bird.kill()
    if bird.y+bird.h > SCREENWIDTH:
        bird.kill()

    bird.move(jumping)
    column1.move()
    column2.move()
    # print(abs(column2.opening.x - column1.opening.x))
    
    if bird.x<=-bird.w:
        end = True
 
    render_game(win, [column1, column2, bird], len(score))
    return end

def render_home(win, bird=None):
    if bird is None:
        bird = Bird()
    win.fill((255,255,255))
    r = START_TEXT.get_rect(center=(SCREENWIDTH//2,SCREENHEIGHT//2))
    win.blit(START_TEXT, r)
    bird.draw(win)
    pygame.display.update()

def homeframe(win, bird=None):
    if bird is None:
        bird = Bird()
    CLOCK.tick(10)
    end = True
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            end = 'shutdown'
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                end = False
    render_home(win, bird)
    return end
    

def game():
    END = True
    bird = None
    score = list()
    column1, column2 = None, None
    max_score = 0
    while True:
        if not END:
            if column1 is None:
                column1 = Column(SCREENWIDTH,Column.WIDTH,Column.OPENING_SIZE)
                column2 = Column(SCREENWIDTH*3//2 + Column.WIDTH//2, Column.WIDTH,Column.OPENING_SIZE)
            END = runframe(window, bird, column1, column2, score)
            if END:
                bird = None
                max_score = max(max_score, len(score))
                print(f'You scored {len(score)}, Highscore: {max_score}')
                score = list()
        else:
            if bird is None:
                bird = Bird()
                column1, column2 = None, None
            END = homeframe(window,bird)
 
        if END == 'shutdown':
            break
    pygame.quit()

if __name__ == "__main__":
    game()
