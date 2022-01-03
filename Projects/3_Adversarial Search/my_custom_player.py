import random

from isolation import DebugState
from sample_players import DataPlayer

BOARD_SIZE = 99  # No. of cells in the board = board size = width * height = 11 * 9 = 99


class AlphaBetaAgent:
    """Class for Alpha-Beta Pruning algorithm with Iterative Deepening Depth First Search. The algorithm will call custom
    heuristic method (needs to be defined) to estimate the values of nodes at each level. Use self.compute() method to run
    the search. This method will yeild the best action computed by the algorithm at each depth level until the max depth
    level has been reached or until the search has been called off. If a terminal state is not encountered at a particular
    level, then self.heuristic() method will give the estimate of the nodes (using heuristic function) at that level,
    otherwise for terminal state, gameState.utility() will give the values."""

    def __init__(self, agent_player, opponent_player, h_type=0):
        self.agent_player = agent_player
        self.opponent_player = opponent_player
        self.h_type = h_type

    def heuristic(self, gameState):
        """There are two types of heuristic defined in this method one for the benchmark agent and one for the custom agent
        (that needs to perform better than the benchmark agent). Initialze the class object with h_type = 0 for getting
        benchmark scores otherwise use h_type = 1 for the custom agent."""
        agent_loc = gameState.locs[self.agent_player]
        opponent_loc = gameState.locs[self.opponent_player]
        agent_liberties = gameState.liberties(agent_loc)
        opponent_liberties = gameState.liberties(opponent_loc)

        # Benchmark Agent Heuristic => #my_moves - #opponent_moves
        if self.h_type == 0:
            return len(agent_liberties) - len(opponent_liberties)

        # Custom Agent Heuristic
        """The custom heuristic, that is developed for the custom agent, will first calculate the value ‘m’, 
        where m = Ply count/Board size, this represents what percentage of the board has been filled. 
        We will use it to determine if the custom agent will play offensively or defensively."""
        if self.h_type == 1:
            m = gameState.ply_count / BOARD_SIZE

            if m < 0.5:
                # If board filled% < 50%, play offensively.
                return len(agent_liberties) - (2 * len(opponent_liberties))
            else:
                # If board filled% > 50%, play defensively.
                return (2 * len(agent_liberties)) - len(opponent_liberties)

    def compute(self, gameState, max_depth):
        """This method will yield the best action computed by the algorithm at each depth level until the max depth
        level has been reached or until the search has been called off. If a terminal state is not encountered at a particular
        level, then self.heuristic() method will give the estimate of the nodes (using heuristic function) at that level,
        otherwise for terminal state, gameState.utility() will give the values."""
        for depth in range(1, max_depth + 1):
            yield self.minimax_decision(gameState, depth)

    def minimax_decision(self, gameState, depth):
        """Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player."""
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None

        for action in gameState.actions():
            value = self.min_value(gameState.result(action), alpha, beta, depth - 1)
            alpha = max(alpha, value)
            if value > best_score:
                best_score = value
                best_move = action

        if best_move == None:
            best_move = random.choice(gameState.actions())

        return best_move

    def min_value(self, gameState, alpha, beta, depth):
        """Return the game state utility if the game is over,
        otherwise return the minimum value over all legal successors."""
        if gameState.terminal_test():
            return gameState.utility(self.agent_player)

        if depth <= 0:
            return self.heuristic(gameState)

        value = float("inf")
        actions = gameState.actions()

        if actions != None:
            for action in actions:
                value = min(value, self.max_value(gameState.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)

        return value

    def max_value(self, gameState, alpha, beta, depth):
        """Return the game state utility if the game is over,
        otherwise return the maximum value over all legal successors."""
        if gameState.terminal_test():
            return gameState.utility(self.agent_player)

        if depth <= 0:
            return self.heuristic(gameState)

        value = float("-inf")
        actions = gameState.actions()

        if actions != None:
            for action in actions:
                value = max(value, self.min_value(gameState.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)

        return value


class CustomPlayer(DataPlayer):
    """Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        """Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        self.agent_player = self.player_id
        self.opponent_player = 1 - self.player_id

        # Use h_type = 0 for the benchmark player, otherwise use h_type = 1 for the custom player.
        player = AlphaBetaAgent(self.agent_player, self.opponent_player, h_type=1)

        # Run the IDDFS Alpha-Beta Pruning algorithm, player.compute() will yield the best action at each depth.
        for action in player.compute(state, max_depth=100):
            self.queue.put(action)
