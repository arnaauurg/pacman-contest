# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions

#game constants (same values as in capture.py)
SIGHT_RANGE = 5     #we see exact enemy positions only within this manhattan distance


##########################
## SEARCH ALGORITHMS #####
##########################

#the search algorithms seen in class: bfs, ucs, a* and minimax with alpha-beta.
#all of them work over a search problem with the same interface as in the course
#(get_start_state, is_goal_state, get_successors)

#the goal of this changes in general is to movee from hardcoded agents (one always attacks, the other
#always defends, with fixed plans like patrolling top/bottom) to more flexible agents that
#decide every turn what to do.

class GridPathProblem:

    """
    Search problem on the maze grid.
    start: cell where we start
    goals: set of goal cells
    allowed: (optional) set of cells the path can use, e.g. the cells we know are safe
    cost_fn: (optional) cost of stepping into a cell, 1 by default
    """

    def __init__(self, neighbors, start, goals, allowed=None, cost_fn=None):
        self.neighbors = neighbors
        self.start = start
        self.goals = set(goals)
        self.allowed = allowed
        self.cost_fn = cost_fn or (lambda c: 1)

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state in self.goals

    def get_successors(self, state):

        """
        Returns a list of (next_state, action, cost) like in the course problems.
        """
        x, y = state
        out = []
        for n in self.neighbors[state]:

            #skip cells outside the allowed region (the goal is always allowed)
            if self.allowed is not None and n not in self.allowed and n not in self.goals:
                continue
            action = Actions.vector_to_direction((n[0] - x, n[1] - y))
            out.append((n, action, self.cost_fn(n)))
        return out


def null_heuristic(state, problem=None):
    return 0


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    
    #track all visited states to avoid repeated points or doing circles
    explored_state = set()
    
    # use stack for lifo order
    frontier_dfs = util.Stack()
    
    # push initial state with empty path and zero cost    
    start_state = [problem.get_start_state(), [], 0]
    frontier_dfs.push(start_state)
       
    # loop until goal found and pacman won
    while (not frontier_dfs.is_empty()):
        
        # pop most recently added state from stack
        [current_state, path, cost] = frontier_dfs.pop()
        
        # only process if not visited before
        if (not current_state in explored_state): 
            explored_state.add(current_state)
            
            # check if we won
            if (problem.is_goal_state(current_state)):
                return path
                
            # get all possible next states
            successors = problem.get_successors(current_state)
            
            # process each child node
            for child in successors:
                [state, action, cost_state] = child
                
                if (not state in explored_state):
                    
                    # we add new action to current path
                    new_path = path + [action]
                    
                    # we update the state
                    new_state = [state, new_path, cost_state]
                    
                    # push child to stack for later exploration
                    frontier_dfs.push(new_state)


def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    
    # set to track visited nodes
    explored_bfs = set()
    frontier_bfs = util.Queue()
    
    # we initialize state and empty path
    start_state = [problem.get_start_state(), [], 0]
    frontier_bfs.push(start_state)
        
    # we continue until the queue is empty 
    while (not frontier_bfs.is_empty()):
        
        # unqueue oldest state from front
        [current_state, path, cost] = frontier_bfs.pop()
        
        # we process only non explored states
        if (not current_state in explored_bfs):
            explored_bfs.add(current_state) 
            
            # return path if the goal is achieved
            if (problem.is_goal_state(current_state)):
                return path
            
            # get all neighbors of current state
            successors = problem.get_successors(current_state)
            
            # iterate for all successors
            for child in successors: 
                [state, action, cost_state] = child
                
                # add only new unexplored states!!
                #before we also checked "not child in frontier_bfs.list" to not repeat states,
                #reepeated states in the queue are harmless: when popped they are already
                #explored and skipped, and bfs still returns the shortest path
                if (not state in explored_bfs):
                    new_path = path + [action]
                    new_state = [state, new_path, cost_state]
                    frontier_bfs.push(new_state)


def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    # set to track explored states
    a_explored = set()
    a_frontier = util.PriorityQueue()
    
    # p.queue follows f = g(n) + h(n)
    start = problem.get_start_state()
    start_node = [start, [], 0]
    a_frontier.push(start_node, 0)
    
    # loop until frontier is empty
    while (not a_frontier.is_empty()):
        
        # get the lowest cost from the queue and pop
        [current_state, path, cost] = a_frontier.pop()
        
        # check if the state has not been explored
        if (not current_state in a_explored):
            a_explored.add(current_state)
            
            # return if goal found
            if (problem.is_goal_state(current_state)):
                return path
            
            # get all successors
            successors = problem.get_successors(current_state)
            
            # for every successor we evaluate            
            for child in successors:
                [state, action, cost_state] = child 
                
                # we check if worth exploring or not
                #same as in bfs: "not child in a_frontier.heap" compared (state, action, cost)
                #with the heap entries (priority, count, item), never equal, and O(n) per child.
                #in a* we even want repeated states in the queue: a later push of the same
                #state can be cheaper, the queue pops the cheapest one first and the
                #explored check when popping throws away the expensive copies
                if (not state in a_explored):
                    
                    # we add the current cost
                    new_cost = cost_state + cost
                    new_path = path + [action]
                    new_state = [state, new_path, new_cost]
                    
                    # we calculate the heuristic value: f(n) = g(n) + h(n)
                    h = new_cost + heuristic(new_state[0], problem)
                    a_frontier.push(new_state, h)


def manhattan_heuristic(state, problem):

    """
    Manhattan distance to the closest goal.
    It is admissible because every step costs at least 1 and walls only make paths longer.
    """
    return min(util.manhattan_distance(state, g) for g in problem.goals)


def uniform_cost_search(problem):

    """
    Search the node of least total cost first.
    Ucs is exactly a* with the null heuristic.
    """
    return a_star_search(problem, null_heuristic)


def alpha_beta(state, depth, alpha, beta, maximizing, successors, evaluate, terminal):

    """
    Depth limited minimax with alpha-beta pruning.
    successors(state, maximizing): list of (move, next_state)
    evaluate(state): value of a non terminal state when we reach the depth limit
    terminal(state): value if the state is terminal (caught, escaped...), else None
    Returns (value, best_move) for the player that moves at this node.
    """
    t = terminal(state)
    if t is not None:
        return t, None
    if depth == 0:
        return evaluate(state), None

    best_move = None

    #max player: our agent
    if maximizing:
        value = float('-inf')
        for move, nxt in successors(state, True):
            v, _ = alpha_beta(nxt, depth - 1, alpha, beta, False, successors, evaluate, terminal)
            if v > value:
                value, best_move = v, move
            alpha = max(alpha, value)

            #min player already has a better option somewhere else, we prune
            if alpha >= beta:
                break

    #min player: the enemy agent
    else:
        value = float('inf')
        for move, nxt in successors(state, False):
            v, _ = alpha_beta(nxt, depth - 1, alpha, beta, True, successors, evaluate, terminal)
            if v < value:
                value, best_move = v, move
            beta = min(beta, value)
            if alpha >= beta:
                break
    return value, best_move


#####################
## TEAM CREATION ####
#####################

def create_team(first_index, second_index, is_red,
                first='AttackAgent', second='DefendAgent', num_training=0):

    """
    This function should return a list of two agents that will form the
    team. One agent attacks and the other one defends.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


####################
## MAP ANALYSIS ####
####################

class MapInfo:

    """
    Static information about the map. We compute it once at the start of the game
    so we dont waste time on it every turn.
    """

    def __init__(self, game_state, red):
        walls = game_state.get_walls()
        self.walls = walls
        self.width, self.height = walls.width, walls.height

        #all free cells and their free neighbours, with the framework helpers walls.as_list and
        #Actions.get_legal_neighbors (computed only once because we use them all the time)
        self.cells = walls.as_list(False)
        cell_set = set(self.cells)
        self.neighbors = {c: Actions.get_legal_neighbors(c, walls) for c in self.cells}

        #our side of the map depends on the team color
        mid = self.width // 2
        self.red = red
        if red:
            self.home_x = mid - 1
            self.is_home = lambda c: c[0] < mid
        else:
            self.home_x = mid
            self.is_home = lambda c: c[0] >= mid

        #boundary = our last column, but only cells that really connect to the enemy side
        self.boundary = [(self.home_x, y) for y in range(self.height) if not walls[self.home_x][y]]
        enemy_x = mid if red else mid - 1
        self.boundary = [b for b in self.boundary if (enemy_x, b[1]) in cell_set] or self.boundary

        #dead ends: we keep removing cells with only one exit (degree <= 1).
        #what gets removed are corridors with no way out, where a ghost can trap us
        degree = {c: len(self.neighbors[c]) for c in self.cells}
        removed = set()
        q = util.Queue()
        for c in self.cells:
            if degree[c] <= 1:
                q.push(c)
        while not q.is_empty():
            c = q.pop()
            if c in removed:
                continue
            removed.add(c)
            for n in self.neighbors[c]:
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] <= 1:
                        q.push(n)

        #for every dead end cell we save how deep it is and where the "mouth" of the corridor is.
        #cells that are not in a dead end have depth 0 and they are their own exit
        self.depth = {c: 0 for c in self.cells}
        self.exit = {c: c for c in self.cells}
        if len(removed) < len(self.cells):

            #bfs starting from all the mouths at the same time, going inside the corridors
            q = util.Queue()
            for c in self.cells:
                if c not in removed:
                    for n in self.neighbors[c]:
                        if n in removed and self.exit[n] == n and self.depth[n] == 0:
                            self.depth[n] = 1
                            self.exit[n] = c
                            q.push(n)
            seen = set(q.list)
            while not q.is_empty():
                c = q.pop()
                for n in self.neighbors[c]:
                    if n in removed and n not in seen:
                        seen.add(n)
                        self.depth[n] = self.depth[c] + 1
                        self.exit[n] = self.exit[c]
                        q.push(n)


#######################
## TEAM TACTICS #######
#######################

class TeamMemory:

    """
    Shared state between team agents for coordination.
    There is one per team color and it is reset every game.
    """

    def __init__(self):
        self.beliefs = {}       #{enemy_index: set of cells where the enemy can be}
        self.last_turn = -1     #timeleft when the memory was created
        self.targets = {}       #{agent_index: target position}, to not chase the same food


_MEMORY = {}


##############
## AGENTS ####
##############

class BaseAgent(CaptureAgent):

    """
    Common code for both agents: enemy tracking and defending.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state):

        """
        Initializes agent and pre-calculates the map info.
        """
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.map = MapInfo(game_state, self.red)

        #new game -> new team memory (timeleft only goes down during a game)
        key = self.red
        mem = _MEMORY.get(key)
        if mem is None or mem.last_turn < game_state.data.timeleft:
            mem = TeamMemory()
            _MEMORY[key] = mem

            #at the start we know exactly where the enemies are
            for e in self.get_opponents(game_state):
                mem.beliefs[e] = {game_state.get_initial_agent_position(e)}
            mem.last_turn = game_state.data.timeleft + 1
        self.mem = mem

        self.enemies = self.get_opponents(game_state)
        self.team = self.get_team(game_state)
        self.total_food = len(self.get_food(game_state).as_list())

        #to detect which of our food gets eaten every turn
        self.prev_def_food = set(self.get_food_you_are_defending(game_state).as_list())
        self.last_eaten = None
        self.last_eaten_turn = -999

        self.turn = 0
        self.guard_cache = None
        self.guard_cache_turn = -999

    #utilities

    def dist(self, a, b):
        return self.get_maze_distance(a, b)

    def my_pos(self, gs):
        p = gs.get_agent_position(self.index)
        return (int(p[0]), int(p[1]))

    def moves_left(self, gs):

        #timeleft counts the moves of the 4 agents, so we have 1/4 of them
        return gs.data.timeleft // 4

    def action_to(self, gs, nxt):

        """
        Returns the action that moves us from our position to the neighbour cell nxt.
        """
        pos = self.my_pos(gs)
        for a in gs.get_legal_actions(self.index):
            dx, dy = Actions.direction_to_vector(a)
            if (int(pos[0] + dx), int(pos[1] + dy)) == nxt:
                return a
        return None

    #enemy tracking

    def update_beliefs(self, gs):

        """
        We keep for every enemy the set of cells where it could be.
        Every turn: the set grows one step (the enemy moved) and then we remove
        the cells that dont match what we observe (noisy distance, sight range,
        pacman or ghost, food that disappeared).
        """
        mem = self.mem
        my_pos = self.my_pos(gs)
        team_pos = [gs.get_agent_position(i) for i in self.team]
        noisy = gs.get_agent_distances()
        prev = self.get_previous_observation()

        #food we defend that disappeared since our last turn -> an invader was there
        cur_food = set(self.get_food_you_are_defending(gs).as_list())
        eaten = self.prev_def_food - cur_food
        self.prev_def_food = cur_food
        if eaten:
            self.last_eaten = min(eaten, key=lambda f: self.dist(my_pos, f))
            self.last_eaten_turn = self.turn

        for e in self.enemies:
            st = gs.get_agent_state(e)
            pos = st.get_position()

            #we see it, so we know exactly where it is
            if pos is not None:
                mem.beliefs[e] = {(int(pos[0]), int(pos[1]))}
                continue

            old = mem.beliefs.get(e) or set(self.map.cells)

            #it was next to one of us last turn and now we dont see it -> probably eaten,
            #so it went back to its start position
            if prev is not None:
                ppos = prev.get_agent_position(e)
                if ppos is not None and any(self.dist(ppos, (int(t[0]), int(t[1]))) <= 1 for t in team_pos):
                    old = {gs.get_initial_agent_position(e)}

            #only the enemy that played just before us has moved since our teammate's turn
            if e == (self.index - 1) % gs.get_num_agents():
                new = set()
                for c in old:
                    new.add(c)
                    new.update(self.map.neighbors[c])
            else:
                new = set(old)

            #remove the cells that dont match the observation
            is_pac = st.is_pacman
            filt = set()
            for c in new:

                #if it is a pacman it is on our side, if it is a ghost on its side
                if self.map.is_home(c) != is_pac:
                    continue

                #if it was that close we would see it
                if any(util.manhattan_distance(c, t) <= SIGHT_RANGE for t in team_pos):
                    continue

                #the real distance must be possible for the noisy reading
                #(framework get_distance_prob gives 0 when it is not)
                if noisy is not None:
                    md = util.manhattan_distance(c, my_pos)
                    if gs.get_distance_prob(md, noisy[e]) == 0:
                        continue
                filt.add(c)

            #food eaten by an invisible invader tells us exactly where it is
            if is_pac and eaten:
                cand = [f for f in eaten if f in filt or f in new]
                if cand:
                    filt = {cand[0]}

            #nothing matches (we made a wrong assumption): start again from the noisy reading only
            if not filt:
                filt = set()
                for c in self.map.cells:
                    if self.map.is_home(c) != is_pac:
                        continue
                    if any(util.manhattan_distance(c, t) <= SIGHT_RANGE for t in team_pos):
                        continue
                    if noisy is not None:
                        md = util.manhattan_distance(c, my_pos)
                        if gs.get_distance_prob(md, noisy[e]) == 0:
                            continue
                    filt.add(c)
                if not filt:
                    filt = {gs.get_initial_agent_position(e)}
            mem.beliefs[e] = filt

    def belief_center(self, e, ref=None):

        """
        Returns one cell that represents where enemy e probably is.
        """
        cells = list(self.mem.beliefs.get(e, ()))
        if not cells:
            return None
        if len(cells) == 1:
            return cells[0]
        if ref is not None:
            return min(cells, key=lambda c: self.dist(ref, c))

        #the cell in the "middle" of the possible cells (we use a sample to save time)
        sample = cells if len(cells) <= 25 else random.sample(cells, 25)
        return min(sample, key=lambda c: sum(self.dist(c, o) for o in sample))

    #defending

    def guard_spot(self, gs):

        """
        Cell on our side, near the border, that is closest to our most exposed food.
        Kind of like a policeman waiting in the neighborhood.
        """

        #we recompute it only every 15 turns, it doesnt change that fast
        if self.guard_cache is not None and self.turn - self.guard_cache_turn < 15:
            return self.guard_cache
        food = self.get_food_you_are_defending(gs).as_list() + self.get_capsules_you_are_defending(gs)
        if not food:
            return self.start

        #exposed food = the half of our food closest to the border
        bnd = self.map.boundary
        exposed = sorted(food, key=lambda f: min(self.dist(f, b) for b in bnd))[:max(4, len(food) // 2)]

        #candidates: our cells at most 3 columns away from the border
        cands = [c for c in self.map.cells if self.map.is_home(c)
                 and abs(c[0] - self.map.home_x) <= 3]
        best = min(cands, key=lambda c: sum(self.dist(c, f) for f in exposed) / len(exposed)
                   + 0.5 * min(self.dist(c, b) for b in bnd))
        self.guard_cache, self.guard_cache_turn = best, self.turn
        return best

    def home_dist_map(self, target):

        """
        Bfs from the target moving only through our side of the map.
        Some maps have parts of our side that only connect through the enemy side,
        so the normal maze distance is not enough to stay home.
        """
        cache = self.__dict__.setdefault('_hd_cache', {})
        if target in cache:
            return cache[target]
        d = {target: 0}
        q = util.Queue()
        q.push(target)
        while not q.is_empty():
            c = q.pop()
            for n in self.map.neighbors[c]:
                if n not in d and self.map.is_home(n):
                    d[n] = d[c] + 1
                    q.push(n)

        #we dont want the cache to grow forever
        if len(cache) > 400:
            cache.clear()
        cache[target] = d
        return d

    def predict_path(self, gs, start, steps=10):

        """
        Predicts the next steps of an invader that goes greedy for the closest food.
        Most attackers do this, so we can wait for it on its way instead of chasing it.
        """
        food = set(self.get_food_you_are_defending(gs).as_list())
        path, cur = [], start
        while len(path) < steps and food:
            f = min(food, key=lambda x: self.dist(cur, x))
            food.discard(f)

            #we walk the shortest path from cur to that food
            while cur != f and len(path) < steps:
                cur = min(self.map.neighbors[cur], key=lambda n: self.dist(n, f))
                path.append(cur)
        return path

    CATCH_MAX_DIST = 12   #use minimax against visible invaders up to this distance
    CATCH_DEPTH = 6        #plies when the invader is close
    CATCH_DEPTH_FAR = 4     #plies when it is further (to save time)

    def minimax_catch(self, gs, pos, inv):

        """
        Minimax with alpha-beta when we see an invader.
        We (max) want to catch it, the invader (min) wants to run back across the border.
        At the depth limit we use territory control: how much of our food and how many
        exits we reach before the invader. We never leave our side.
        """
        is_home = self.map.is_home
        nb = self.map.neighbors
        exits = self.map.boundary

        #a state is (our position, invader position)
        def successors(state, maximizing):
            me, p = state
            if maximizing:
                moves = [n for n in nb[me] if is_home(n)] + [me]
                return [(n, (n, p)) for n in moves]
            return [(n, (me, n)) for n in nb[p] + [p]]

        def terminal(state):
            me, p = state
            if me == p:
                return 1000     #we caught it
            if not is_home(p):
                return -300     #it escaped to its side
            return None

        #territory control: food we reach before the invader is food it cant take safely,
        #and border cells we reach first are exits we keep closed
        food = self.get_food_you_are_defending(gs).as_list() + self.get_capsules_you_are_defending(gs)

        def evaluate(state):
            me, p = state
            ctrl = sum(1 for f in food if self.dist(me, f) < self.dist(p, f))
            cut = sum(1 for b in exits if self.dist(me, b) < self.dist(p, b))
            return 10 * ctrl + 25 * cut / max(1, len(exits)) - self.dist(me, p)

        depth = self.CATCH_DEPTH if self.dist(pos, inv) <= 4 else self.CATCH_DEPTH_FAR
        _, move = alpha_beta((pos, inv), depth, float('-inf'), float('inf'),
                             True, successors, evaluate, terminal)
        if move is None:
            return None
        if move == pos:
            return Directions.STOP
        return self.action_to(gs, move)

    def defend_action(self, gs, stay_home=True):

        """
        Main defensive decision.
        Priority: catch a close invader (minimax) > intercept a visible invader >
        go where food was just eaten > go where we think the invader is > guard spot.
        """
        pos = self.my_pos(gs)
        my_state = gs.get_agent_state(self.index)
        scared = my_state.scared_timer > 0
        legal = [a for a in gs.get_legal_actions(self.index) if a != Directions.STOP]

        #invaders = enemies that are pacman on our side
        invaders = [e for e in self.enemies if gs.get_agent_state(e).is_pacman]
        visible = [e for e in invaders if gs.get_agent_state(e).get_position() is not None]

        #invader very close and we are not scared -> adversarial search
        if visible and not scared:
            e = min(visible, key=lambda e: self.dist(pos, self.belief_center(e)))
            ip = self.belief_center(e)
            if self.dist(pos, ip) <= self.CATCH_MAX_DIST:
                a = self.minimax_catch(gs, pos, ip)
                if a is not None:
                    return a

        #choose the target position
        target = None
        inv_pos = None
        if visible:
            e = min(visible, key=lambda e: self.dist(pos, self.belief_center(e)))
            inv_pos = self.belief_center(e)
            target = inv_pos

            #chasing from behind doesnt work (same speed), so we go to the first
            #cell of its predicted path where we arrive before or at the same time
            if not scared and self.dist(pos, inv_pos) > 2:
                for k, c in enumerate(self.predict_path(gs, inv_pos), start=1):
                    if self.dist(pos, c) <= k:
                        target = c
                        break
        elif self.last_eaten is not None and self.turn - self.last_eaten_turn <= 6:
            #we dont see it but some food was eaten recently
            target = self.last_eaten
        elif invaders:
            #we dont see it, we use the most precise belief
            e = min(invaders, key=lambda e: len(self.mem.beliefs.get(e, ())))
            target = self.belief_center(e)
        if target is None:
            target = self.guard_spot(gs)

        #if we can, we move only through our side of the map
        hd = self.home_dist_map(target) if (stay_home and self.map.is_home(target)) else None
        use_home = hd is not None and pos in hd

        #we score every action, lower is better
        options = []
        for a in legal:

            #cell where we end up with this action (framework helper)
            x, y = Actions.get_successor(pos, a)
            npos = (int(x), int(y))
            if use_home:
                if npos not in hd:
                    continue
                d = hd[npos]
            else:
                d = self.dist(npos, target)

            #if we are scared we cant eat it: we follow it keeping a distance of 2
            if scared and inv_pos is not None:
                di = self.dist(npos, inv_pos)
                score = abs(di - 2) + (100 if di <= 1 else 0)
            else:
                score = d
            options.append((score, random.random(), a))  #random to break ties
        if not options:
            return Directions.STOP
        options.sort()

        #already at the guard spot and nobody around -> wait there
        if inv_pos is None and pos == target:
            return Directions.STOP
        return options[0][2]


class AttackAgent(BaseAgent):

    """
    Offensive agent.
    Every turn:
    1. bfs to know which cells we reach before any ghost (the "safe region")
    2. decide the target: home, capsule, scared ghost or the closest safe food
    3. a* inside the safe region to go to the target
    4. if there is no safe path at all, minimax to escape from the ghost
    """

    SAFETY = 1          #extra margin (in moves) we want against ghosts (-1 when carrying nothing)
    MINIMAX_DEPTH = 6   #plies for the escape minimax (3 moves each)

    def dangerous_ghosts(self, gs, pos):

        """
        Visible enemy ghosts that can eat us (not scared, or scared but about to wake up).
        """
        res = []
        for e in self.enemies:
            st = gs.get_agent_state(e)
            p = st.get_position()
            if p is None or st.is_pacman:
                continue
            p = (int(p[0]), int(p[1]))

            #scared for longer than it takes us to get there -> not dangerous
            if st.scared_timer > self.dist(pos, p) + 2:
                continue
            res.append(p)
        return res

    def safe_search(self, gs, start, ghosts):

        """
        Bfs from our position that only goes through cells we reach strictly before
        any dangerous ghost. Returns the distances, the parents and the ghost distance function.
        """
        gd_cache = {}

        #gd(c) = distance from the closest ghost to cell c
        def gd(c):
            if c not in gd_cache:
                gd_cache[c] = min((self.dist(g, c) for g in ghosts), default=999)
            return gd_cache[c]

        dist = {start: 0}
        parent = {start: None}
        q = util.Queue()
        q.push(start)
        while not q.is_empty():
            c = q.pop()
            t = dist[c]
            for n in self.map.neighbors[c]:
                if n in dist:
                    continue

                #on our side ghosts cant eat us. On their side we must arrive at n
                #(at time t+1) before the ghost can
                if not self.map.is_home(n) and gd(n) <= t + self.SAFETY:
                    continue
                if not self.map.is_home(n) and n in ghosts:
                    continue
                dist[n] = t + 1
                parent[n] = c
                q.push(n)
        return dist, parent, gd

    def first_step(self, parent, target):

        """
        Walks back the bfs parents to get the first cell of the path to target.
        """
        c = target
        while parent[c] is not None and parent[parent[c]] is not None:
            c = parent[c]
        return c

    def is_trap(self, cell, t, gd):

        """
        True if going to cell (we arrive at time t) can get us trapped in a dead end:
        the ghost reaches the mouth of the corridor before we get out.
        """
        d = self.map.depth[cell]
        if d == 0:
            return False
        mouth = self.map.exit[cell]
        return gd(mouth) <= t + d + self.SAFETY

    def choose_action(self, gs):

        """
        Main offensive decision function.
        """
        self.turn += 1
        self.update_beliefs(gs)
        pos = self.my_pos(gs)
        st = gs.get_agent_state(self.index)
        carrying = st.num_carrying
        ghosts = self.dangerous_ghosts(gs, pos)

        #with no food on us dying only costs the walk back, so we take more risks
        #(we accept that the ghost could get one step closer than normally)
        self.SAFETY = -1 if carrying == 0 else 1

        #1. safe region with bfs
        dist, parent, gd = self.safe_search(gs, pos, ghosts)

        food = self.get_food(gs).as_list()
        caps = self.get_capsules(gs)
        moves_left = self.moves_left(gs)
        home_cells = [b for b in self.map.boundary if b in dist]
        true_home = min(self.dist(pos, b) for b in self.map.boundary)
        score = self.get_score(gs)

        #end of the game and we are winning: stay home and defend the lead
        if st.is_pacman is False and score > 0 and moves_left < 40:
            return self.defend_action(gs)

        #invader close to us while we are still on our side: we eat it first
        if not st.is_pacman and st.scared_timer == 0:
            for e in self.enemies:
                es = gs.get_agent_state(e)
                ep = es.get_position()
                if es.is_pacman and ep is not None and self.dist(pos, ep) <= 4:
                    return self.defend_action(gs)

        ghost_near = min((self.dist(pos, g) for g in ghosts), default=999)

        #nobody defending their side (both enemies are pacman): we can carry more food
        undefended = all(gs.get_agent_state(e).is_pacman for e in self.enemies) and not ghosts

        #2. should we go home to cash the food?
        go_home = False
        if carrying > 0:
            if moves_left <= true_home + 4:
                go_home = True      #game is ending, food carried is worth nothing
            elif len(food) <= 2:
                go_home = True      #we already ate all the food we need
            elif ghost_near <= 5 and carrying >= 2:
                go_home = True      #ghost close, dont risk what we have
            elif carrying >= 8 and not undefended:
                go_home = True      #carrying a lot
            elif carrying >= 4 and true_home <= 3 and not undefended:
                go_home = True      #home is very close, cheap to cash
            elif score + carrying >= self.total_food - 2:
                go_home = True      #cashing this wins the game

        #targets we reach safely and that are not a trap
        def safe_targets(cells):
            out = []
            for c in cells:
                if c in dist and not self.is_trap(c, dist[c], gd):
                    out.append((dist[c], c))
            return out

        target = None
        if go_home:
            if home_cells:
                target = min(home_cells, key=lambda b: dist[b])

            #if home is cut off (or a ghost is on us) a capsule can save us
            if target is None or (ghost_near <= 3 and caps):
                sc = safe_targets(caps)
                if sc and (target is None or min(sc)[0] < dist[target]):
                    target = min(sc)[1]
        else:
            #ghost near: take a close safe capsule
            if ghosts and ghost_near <= 5:
                sc = safe_targets(caps)
                if sc and min(sc)[0] <= 8:
                    target = min(sc)[1]

            #scared ghost very close: we eat it
            if target is None:
                for e in self.enemies:
                    es = gs.get_agent_state(e)
                    ep = es.get_position()
                    if ep is None or es.is_pacman:
                        continue
                    ep = (int(ep[0]), int(ep[1]))
                    if ep in dist and es.scared_timer > dist[ep] + 1 and dist[ep] <= 3:
                        target = ep

            #normal case: closest safe food
            if target is None:
                sf = safe_targets(food)
                if sf:

                    #avoid food that teammate is targeting (when it also attacks)
                    mate_t = [t for i, t in self.mem.targets.items() if i != self.index and t]
                    if mate_t and len(sf) > 1:
                        sf = [x for x in sf if all(self.dist(x[1], m) > 3 for m in mate_t)] or sf
                    target = min(sf)[1]
                elif carrying > 0 and home_cells:
                    target = min(home_cells, key=lambda b: dist[b])
        self.mem.targets[self.index] = target

        #3. go to the target with a* inside the safe region
        if target is not None and target != pos:

            #every step costs 1, plus a penalty when the cell is close to a ghost,
            #so between two safe paths we take the one further from ghosts
            def cost(c):
                if not ghosts or self.map.is_home(c):
                    return 1
                return 1 + 8.0 / max(1, gd(c))
            problem = GridPathProblem(self.map.neighbors, pos, [target], allowed=dist, cost_fn=cost)
            path = a_star_search(problem, manhattan_heuristic)
            if path and path[0] in gs.get_legal_actions(self.index):
                return path[0]

            #fallback: first step of the bfs path
            nxt = self.first_step(parent, target)
            a = self.action_to(gs, nxt)
            if a is not None:
                return a

        #4. no safe target
        #at home: move along the border to find an entrance without ghosts
        if not st.is_pacman:
            return self.reposition(gs, pos, ghosts)

        #on their side: minimax against the ghost
        if ghosts:
            a = self.minimax_escape(gs, pos, ghosts, carrying)
            if a is not None:
                return a
        return self.evade(gs, pos, ghosts)

    def minimax_escape(self, gs, pos, ghosts, carrying):

        """
        Minimax with alpha-beta when we are in danger.
        We (max) want to survive, get home or eat a capsule.
        The closest ghost (min) wants to catch us, and it cant leave its side.
        """
        ghost = min(ghosts, key=lambda g: self.dist(pos, g))
        caps = set(self.get_capsules(gs))
        is_home = self.map.is_home
        nb = self.map.neighbors
        bnd = self.map.boundary

        #a state is (our position, ghost position)
        def successors(state, maximizing):
            me, g = state
            if maximizing:
                return [(n, (n, g)) for n in nb[me]]
            moves = [n for n in nb[g] if not is_home(n)] + [g]
            return [(n, (me, n)) for n in moves]

        def terminal(state):
            me, g = state
            if me == g:
                return -1000                #caught
            if is_home(me):
                return 500 + 10 * carrying  #home safe, more food = better
            if me in caps:
                return 400                  #capsule, the ghost gets scared
            return None

        def evaluate(state):

            #far from the ghost, out of dead ends and close to home is better
            me, g = state
            dg = self.dist(me, g)
            dh = min(self.dist(me, b) for b in bnd)
            return 6 * min(dg, 6) - 4 * self.map.depth[me] - 1.5 * dh

        _, move = alpha_beta((pos, ghost), self.MINIMAX_DEPTH, float('-inf'), float('inf'),
                             True, successors, evaluate, terminal)
        return self.action_to(gs, move) if move is not None else None

    def reposition(self, gs, pos, ghosts):

        """
        We are home and every entrance looks guarded.
        We move to the border cell furthest from the ghosts to try to enter there.
        """
        if ghosts:
            entry = max(self.map.boundary,
                        key=lambda b: min(self.dist(b, g) for g in ghosts) - 0.3 * self.dist(pos, b))
        else:
            entry = random.choice(self.map.boundary)

        #bfs only through our side, we dont want to cross while moving
        home = {c for c in self.map.cells if self.map.is_home(c)}
        path = breadth_first_search(GridPathProblem(self.map.neighbors, pos, [entry], allowed=home))
        if path and path[0] in gs.get_legal_actions(self.index):
            return path[0]
        return Directions.STOP

    def evade(self, gs, pos, ghosts):

        """
        Worst case scenario: no safe path and minimax gave nothing.
        We take the action that keeps us far from ghosts, out of dead ends and close to home.
        """
        best, bs = Directions.STOP, -9999
        for a in gs.get_legal_actions(self.index):
            succ = gs.generate_successor(self.index, a)
            np_ = succ.get_agent_position(self.index)
            np_ = (int(np_[0]), int(np_[1]))
            g = min((self.dist(np_, gp) for gp in ghosts), default=20)
            s = 3 * min(g, 6) - 2 * self.map.depth[np_] - 0.5 * min(self.dist(np_, b) for b in self.map.boundary)
            if g <= 1:
                s -= 100
            if self.map.is_home(np_):
                s += 50
            if s > bs:
                best, bs = a, s
        return best


class DefendAgent(BaseAgent):

    """
    Defensive agent. It always stays on our side (see defend_action in BaseAgent).
    """

    def choose_action(self, gs):
        self.turn += 1
        self.update_beliefs(gs)
        return self.defend_action(gs)
