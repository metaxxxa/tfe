import os
def write_terrain(dir, name, terrain):
    """Writes representation of terrain to file with
    filename 'name.ter'

    Args:
        name ([str]): name of terrain
        terrain ([list]): terrain object as list; elements are coordinates of objects
    """
    size = terrain['size']
    s = render_terrain(terrain)
    if not os.path.exists(f'env/terrains/{dir}'):
            os.makedirs(f'env/terrains/{dir}')
    with open(f'env/terrains/{dir}/{name}_{size}x{size}.ter', 'w') as f:
        f.write(s)

def load_terrain(name):
    terrain = {'obstacles': [], 'blue': [], 'red': []}
    with open('env/terrains/' + name + '.ter', 'r') as f:
    #with open('terrains/' + name + '.ter', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                terrain['size'] = len(line)-1
            for key, ident in [('obstacles', 'x'), ('blue', '1'), ('red', '2')]:
                for n in [n for n in range(len(line)) if line.find(ident, n) == n]: # all occurences of 'x'
                    terrain[key].append((idx, n))
    return terrain

def make_terrain(size):
    terrain = {'size': size,
                'obstacles': []}
    if isinstance(size, int):
        s0, s1 = size, size
    elif isinstance(size, tuple):
        s0, s1 = size
    terrain['obstacles'].append((s0//2-1, s1//2-1))
    terrain['obstacles'].append((s0//2,   s1//2-1))
    terrain['obstacles'].append((s0//2-1, s1//2))
    terrain['obstacles'].append((s0//2,   s1//2))
    return terrain

def render_terrain(terrain):
    size = terrain['size']
    obstacles = terrain['obstacles']
    blue = []
    red = []
    if 'blue' in terrain.keys():
        blue = terrain['blue']
        red = terrain['red']
    s = ''
    for x in range(size):
        for y in range(size):
            if (x, y) in obstacles:
                s += 'x'
            elif (x, y) in blue:
                s += '1'
            elif (x, y) in red:
                s += '2'
            else:
                s += '.'
        s += '\n'
    return s

if __name__ == '__main__':
    terrain = make_terrain(10)
    print(terrain)
    write_terrain('','central', terrain)
    print(load_terrain('central_10x10'))
