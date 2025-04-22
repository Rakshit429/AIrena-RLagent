import numpy as np
from agents.agent import Agent
import time

class StrategicSacrificeAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Strategic Agent"
        self.time_limit = 90  # Total time limit in seconds
        self.time_used = 0    # Track total time used
        self.start_time = 0   # Track start time of each move
        
        # Directions for virtual connections
        self.directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
        
        # Determine axis of play based on player number
        if self.player_number == 1:  # top to bottom
            self.my_axis = 1
            self.opp_axis = 0
        else:  # left to right
            self.my_axis = 0
            self.opp_axis = 1
            
        # Initialize pattern database for virtual connections
        self.virtual_pattern_db = self._init_virtual_patterns()
        
        # Initialize history for opponent patterns
        self.opponent_history = []
        
        # Initialize move cache to avoid recalculation
        self.move_cache = {}
        
    def _init_virtual_patterns(self):
        """Initialize patterns for identifying virtual connections"""
        patterns = []
        
        # Bridge pattern: two stones separated by an empty cell
        patterns.append({"pattern": [self.player_number, 0, self.player_number], 
                        "value": 10, 
                        "type": "bridge"})
        
        # Ladder pattern: potential to create a forced path
        patterns.append({"pattern": [self.player_number, 0, 0, self.player_number], 
                        "value": 15, 
                        "type": "ladder"})
        
        # Defensive pattern: block opponent's potential bridge
        patterns.append({"pattern": [self.adv_number, 0, self.adv_number], 
                        "value": 20, 
                        "type": "block"})
        
        return patterns
    
    def step(self):
        """Make a strategic move, considering virtual connections and sacrifices"""
        self.start_time = time.time()
        
        # Check if we need to use optimized calculation due to time constraints
        remaining_time = 90 - self.time_used
        depth = 2
        
        if remaining_time < 30:
            # If less than 30 seconds remain, reduce search depth
            depth = 1
        
        # Get available moves
        available_moves = self.free_moves()
        
        if not available_moves:
            return [0, 0]  # Fallback
        
        # Check if it's the first move - consider strategic openings
        if sum(1 for x in range(self.size) for y in range(self.size) 
               if self.get_hex([x, y]) != 0) <= 1:
            return self._make_opening_move()
        
        # First, look for winning moves
        winning_move = self._find_winning_move()
        if winning_move:
            self._update_time()
            self.set_hex(self.player_number, winning_move)
            return winning_move
        
        # Look for crucial defensive moves (prevent opponent from winning)
        blocking_move = self._find_blocking_move()
        if blocking_move:
            self._update_time()
            self.set_hex(self.player_number, blocking_move)
            return blocking_move
            
        # Calculate move values using minimax with alpha-beta pruning
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in available_moves:
            # Check time before evaluating each move
            if time.time() - self.start_time > 1.0:  # Time guard
                if best_move:  # If we already have a decent move, use it
                    break
            
            # Make temporary move
            new_board = self.copy()
            new_board.set_hex(self.player_number, move)
            
            # Calculate move value
            move_value = self._minimax(new_board, depth, alpha, beta, False)
            
            if move_value > best_value:
                best_value = move_value
                best_move = move
                
            alpha = max(alpha, best_value)
            
            # Check time constraints again
            if time.time() - self.start_time > 1.5:
                break
                
        # If still no good move found, use strategic pattern-based move
        if not best_move:
            best_move = self._find_strategic_move(available_moves)
            
        self._update_time()
        self.set_hex(self.player_number, best_move)
        return best_move
        
    def _make_opening_move(self):
        """Make a strategic opening move"""
        # If first player, consider center or slightly off-center position
        if self.player_number == 1:
            # For odd-sized boards, center is often good
            if self.size % 2 == 1:
                center = self.size // 2
                return [center, center]
            else:
                # For even-sized boards, slightly off-center
                center = self.size // 2
                return [center, center - 1]
        else:
            # For second player, consider response based on first player's move
            # Find first player's move
            for x in range(self.size):
                for y in range(self.size):
                    if self.get_hex([x, y]) == self.adv_number:
                        # Respond nearby but not adjacent
                        neighbors = self.neighbors([x, y])
                        extended_neighbors = []
                        
                        # Get neighbors of neighbors
                        for n in neighbors:
                            extended_neighbors.extend(self.neighbors(n))
                            
                        # Filter to empty cells that aren't direct neighbors
                        candidates = [n for n in extended_neighbors 
                                    if n not in neighbors and self.get_hex(n) == 0]
                        
                        if candidates:
                            # Select one that improves our position on our axis
                            for candidate in candidates:
                                if (self.player_number == 1 and candidate[1] < y) or \
                                   (self.player_number == 2 and candidate[0] > x):
                                    return candidate
                            
                            # If no directional advantage, pick any
                            return candidates[0]
            
            # If can't find first player's move, use center
            center = self.size // 2
            return [center, center]
    
    def _find_winning_move(self):
        """Find a move that immediately wins the game"""
        moves = self.free_moves()
        
        for move in moves:
            # Try the move
            test_board = self.copy()
            test_board.set_hex(self.player_number, move)
            
            # Check if it's a winning move
            if test_board.check_win(self.player_number):
                return move
                
        return None
    
    def _find_blocking_move(self):
        """Find a move that blocks opponent from winning next turn"""
        moves = self.free_moves()
        
        for move in moves:
            # Try giving this move to opponent
            test_board = self.copy()
            test_board.set_hex(self.adv_number, move)
            
            # Check if opponent would win with this move
            if test_board.check_win(self.adv_number):
                return move
                
        return None
        
    def _find_strategic_move(self, available_moves):
        """Find a move based on strategic patterns and board position"""
        move_values = {}
        
        # Calculate base values for all moves
        for move in available_moves:
            value = self._evaluate_move(move)
            move_values[tuple(move)] = value
            
        # Find bridges and virtual connections
        for move in available_moves:
            # Check if this move creates virtual connections
            value_bonus = self._check_virtual_connections(move)
            move_values[tuple(move)] += value_bonus
            
            # Add bonus for moves that progress toward goal
            progress_value = self._calculate_progress_value(move)
            move_values[tuple(move)] += progress_value
            
        # Find the best move based on values
        best_move = max(available_moves, key=lambda m: move_values[tuple(m)])
        return best_move
    
    def _evaluate_move(self, move):
        """Evaluate the value of a potential move"""
        value = 0
        
        # Prefer moves that extend our territory toward our goal
        if self.player_number == 1:  # Top-bottom player
            # Value increases as we get closer to the middle rows
            mid_row = self.size // 2
            distance_to_mid = abs(move[1] - mid_row)
            value += 10 - distance_to_mid
        else:  # Left-right player
            # Value increases as we get closer to the middle columns
            mid_col = self.size // 2
            distance_to_mid = abs(move[0] - mid_col)
            value += 10 - distance_to_mid
            
        # Check connectivity - more connected neighbors are valuable
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.player_number:
                value += 5
            elif self.get_hex(neighbor) == self.adv_number:
                value += 2  # Smaller bonus for blocking opponent
                
        # Strategic value for edges
        if self.player_number == 1:  # Top-bottom player
            if move[1] == 0 or move[1] == self.size - 1:
                value += 15  # High value for reaching our target edges
        else:  # Left-right player
            if move[0] == 0 or move[0] == self.size - 1:
                value += 15  # High value for reaching our target edges
                
        return value
    
    def _check_virtual_connections(self, move):
        """Check if move creates or extends virtual connections"""
        value = 0
        temp_board = self.copy()
        temp_board.set_hex(self.player_number, move)
        
        # Check in all directions for patterns
        for direction in self.directions:
            path = []
            curr = move.copy()
            
            # Look ahead up to 3 cells in this direction
            for _ in range(3):
                curr = [curr[0] + direction[0], curr[1] + direction[1]]
                if 0 <= curr[0] < self.size and 0 <= curr[1] < self.size:
                    path.append(temp_board.get_hex(curr))
                else:
                    break
                    
            # Check if this path matches any of our patterns
            for pattern in self.virtual_pattern_db:
                if self._is_pattern_match(path, pattern["pattern"]):
                    value += pattern["value"]
                    
        return value
    
    def _is_pattern_match(self, path, pattern):
        """Check if a path matches a pattern"""
        if len(path) < len(pattern):
            return False
            
        # Check if the beginning of the path matches the pattern
        return path[:len(pattern)] == pattern
    
    def _calculate_progress_value(self, move):
        """Calculate how much a move progresses toward our goal"""
        value = 0
        
        if self.player_number == 1:  # Top-bottom connection
            # Check if this move helps connect top to bottom
            # Value moves that create north-south connections
            neighbors = self.neighbors(move)
            north_connection = False
            south_connection = False
            
            for neighbor in neighbors:
                if self.get_hex(neighbor) == self.player_number:
                    if neighbor[1] < move[1]:  # Neighbor is to the north
                        north_connection = True
                    elif neighbor[1] > move[1]:  # Neighbor is to the south
                        south_connection = True
                        
            if north_connection and south_connection:
                value += 30  # High value for moves that connect north and south
                
        else:  # Left-right connection
            # Check if this move helps connect left to right
            # Value moves that create east-west connections
            neighbors = self.neighbors(move)
            west_connection = False
            east_connection = False
            
            for neighbor in neighbors:
                if self.get_hex(neighbor) == self.player_number:
                    if neighbor[0] < move[0]:  # Neighbor is to the west
                        west_connection = True
                    elif neighbor[0] > move[0]:  # Neighbor is to the east
                        east_connection = True
                        
            if west_connection and east_connection:
                value += 30  # High value for moves that connect east and west
                
        return value
    
    def _minimax(self, board, depth, alpha, beta, is_maximizing):
        """Minimax implementation with alpha-beta pruning"""
        # Check for timeout to prevent going over time limit
        if time.time() - self.start_time > 1.0:
            if is_maximizing:
                return float('-inf')  # Return worst value to indicate time problem
            else:
                return float('inf')   # Return worst value to indicate time problem
        
        # Check if board state is cached
        board_hash = hash(str(board._grid))
        if (board_hash, depth, is_maximizing) in self.move_cache:
            return self.move_cache[(board_hash, depth, is_maximizing)]
            
        # Check terminal conditions
        if board.check_win(self.player_number):
            return 1000 + depth  # Winning position, prefer winning sooner
            
        if board.check_win(self.adv_number):
            return -1000 - depth  # Losing position, prefer losing later
            
        if depth == 0:
            # Evaluate board at leaf nodes
            score = self._evaluate_board(board)
            self.move_cache[(board_hash, depth, is_maximizing)] = score
            return score
            
        available_moves = board.free_moves()
        if not available_moves:
            # No moves left - it's a draw
            return 0
            
        if is_maximizing:
            value = float('-inf')
            
            for move in available_moves:
                new_board = board.copy()
                new_board.set_hex(self.player_number, move)
                value = max(value, self._minimax(new_board, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                
                if beta <= alpha:
                    break  # Beta cutoff
                    
            self.move_cache[(board_hash, depth, is_maximizing)] = value
            return value
            
        else:
            value = float('inf')
            
            for move in available_moves:
                new_board = board.copy()
                new_board.set_hex(self.adv_number, move)
                value = min(value, self._minimax(new_board, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                
                if beta <= alpha:
                    break  # Alpha cutoff
                    
            self.move_cache[(board_hash, depth, is_maximizing)] = value
            return value
    
    def _evaluate_board(self, board):
        """Evaluate board position, considering strategic sacrifices"""
        score = 0
        
        # Count cells owned by each player
        my_cells = 0
        adv_cells = 0
        
        for x in range(board.get_grid_size()):
            for y in range(board.get_grid_size()):
                cell = board.get_hex([x, y])
                if cell == self.player_number:
                    my_cells += 1
                elif cell == self.adv_number:
                    adv_cells += 1
                    
        # Slight advantage for having more cells
        score += (my_cells - adv_cells) * 2
        
        # Find shortest paths for both players
        my_path_length = self._shortest_path_length(board, self.player_number)
        adv_path_length = self._shortest_path_length(board, self.adv_number)
        
        # Value having a shorter path to goal
        if my_path_length == float('inf'):
            score -= 500  # Severely penalize if no path exists
        else:
            score += (1000 // (my_path_length + 1))  # More points for shorter paths
            
        if adv_path_length == float('inf'):
            score += 300  # Bonus if opponent has no path
        else:
            score -= (800 // (adv_path_length + 1))  # Penalize short opponent paths
            
        # Value territory control near the middle for both connection axes
        mid_control_value = self._evaluate_middle_control(board)
        score += mid_control_value
        
        # Value having connected components heading toward goal
        connectivity_value = self._evaluate_connectivity(board)
        score += connectivity_value
        
        return score
    
    def _shortest_path_length(self, board, player):
        """Estimate length of shortest path to connect sides for a player"""
        if player == 1:  # Top-bottom connection
            # Check from every cell in top row
            min_length = float('inf')
            
            for x in range(board.get_grid_size()):
                if board.get_hex([x, 0]) == player:
                    path_length = self._bfs_path_length(board, [x, 0], player, 1)
                    min_length = min(min_length, path_length)
                    
            return min_length
        else:  # Left-right connection
            # Check from every cell in leftmost column
            min_length = float('inf')
            
            for y in range(board.get_grid_size()):
                if board.get_hex([0, y]) == player:
                    path_length = self._bfs_path_length(board, [0, y], player, 0)
                    min_length = min(min_length, path_length)
                    
            return min_length
    
    def _bfs_path_length(self, board, start, player, axis):
        """BFS to find shortest path length to goal side"""
        from collections import deque
        
        queue = deque([(start, 0)])  # (position, distance)
        visited = set([tuple(start)])
        target_value = board.get_grid_size() - 1
        
        while queue:
            (x, y), dist = queue.popleft()
            
            # Check if reached goal side
            if (axis == 0 and x == target_value) or (axis == 1 and y == target_value):
                return dist
                
            # Check neighbors
            for neighbor in board.neighbors([x, y]):
                nx, ny = neighbor
                if tuple(neighbor) not in visited and board.get_hex(neighbor) == player:
                    visited.add(tuple(neighbor))
                    queue.append((neighbor, dist + 1))
                    
        return float('inf')  # No path found
    
    def _evaluate_middle_control(self, board):
        """Evaluate control of middle area which is strategically important"""
        score = 0
        size = board.get_grid_size()
        middle_start = size // 3
        middle_end = size - middle_start
        
        # Check middle area of board
        for x in range(middle_start, middle_end):
            for y in range(middle_start, middle_end):
                cell = board.get_hex([x, y])
                if cell == self.player_number:
                    score += 5
                elif cell == self.adv_number:
                    score -= 5
                    
        return score
    
    def _evaluate_connectivity(self, board):
        """Evaluate connectivity of stones toward goal"""
        score = 0
        size = board.get_grid_size()
        
        # Find connected components for the player
        components = self._find_connected_components(board, self.player_number)
        
        if self.player_number == 1:  # Top-bottom player
            # Value components that span multiple rows
            for component in components:
                min_y = min(pos[1] for pos in component)
                max_y = max(pos[1] for pos in component)
                span = max_y - min_y
                
                # More points for components that span more rows
                score += span * 3
                
                # Extra points for components that touch both edges
                if min_y == 0 and max_y == size - 1:
                    score += 100
                elif min_y == 0 or max_y == size - 1:
                    score += 20
                    
        else:  # Left-right player
            # Value components that span multiple columns
            for component in components:
                min_x = min(pos[0] for pos in component)
                max_x = max(pos[0] for pos in component)
                span = max_x - min_x
                
                # More points for components that span more columns
                score += span * 3
                
                # Extra points for components that touch both edges
                if min_x == 0 and max_x == size - 1:
                    score += 100
                elif min_x == 0 or max_x == size - 1:
                    score += 20
                    
        return score
    
    def _find_connected_components(self, board, player):
        """Find all connected components for a player"""
        size = board.get_grid_size()
        visited = set()
        components = []
        
        for x in range(size):
            for y in range(size):
                if board.get_hex([x, y]) == player and (x, y) not in visited:
                    # Start a new component
                    component = []
                    self._dfs_component(board, [x, y], player, visited, component)
                    components.append(component)
                    
        return components
    
    def _dfs_component(self, board, pos, player, visited, component):
        """DFS to find a connected component"""
        x, y = pos
        visited.add((x, y))
        component.append((x, y))
        
        for neighbor in board.neighbors(pos):
            nx, ny = neighbor
            if (nx, ny) not in visited and board.get_hex(neighbor) == player:
                self._dfs_component(board, neighbor, player, visited, component)
    
    def update(self, move_other_player):
        """Update internal state after opponent's move"""
        self.set_hex(self.adv_number, move_other_player)
        
        # Track opponent's moves to identify patterns
        self.opponent_history.append(move_other_player)
        
        # Clear cache since board has changed
        self.move_cache = {}
    
    def _update_time(self):
        """Update the time used for this move"""
        move_time = time.time() - self.start_time
        self.time_used += move_time