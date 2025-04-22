import numpy as np
import time
from agents.agent import Agent

class EdgeControlAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Edge Controller"
        self.time_limit = 90  # Total time limit in seconds
        self.time_used = 0    # Track total time used
        
        # Define edges based on player number
        if player_number == 1:  # top-bottom connection
            self.my_edges = [(x, 0) for x in range(size)] + [(x, size-1) for x in range(size)]
            self.my_primary_axis = 1  # y-axis
            self.adv_primary_axis = 0  # x-axis
        else:  # left-right connection
            self.my_edges = [(0, y) for y in range(size)] + [(size-1, y) for y in range(size)]
            self.my_primary_axis = 0  # x-axis
            self.adv_primary_axis = 1  # y-axis
            
        # Initialize board analysis values
        self.center = size // 2
        self.edge_distance_cache = {}
        self.shortest_path_cache = {}
        self.board_states_seen = {}
        
    def step(self):
        """Execute a move focusing on edge control strategy"""
        start_time = time.time()
        
        # First check for winning moves
        winning_move = self._find_winning_move()
        if winning_move:
            self.set_hex(self.player_number, winning_move)
            self._update_time(start_time)
            return winning_move
            
        # Then check for defensive moves (blocking opponent's win)
        blocking_move = self._find_blocking_move()
        if blocking_move:
            self.set_hex(self.player_number, blocking_move)
            self._update_time(start_time)
            return blocking_move
            
        # Handle first few moves with special opening strategy
        total_stones = sum(1 for x in range(self.size) for y in range(self.size) 
                         if self.get_hex([x, y]) != 0)
        if total_stones <= 1:
            opening_move = self._opening_strategy()
            self.set_hex(self.player_number, opening_move)
            self._update_time(start_time)
            return opening_move
        
        # Get available moves
        moves = self.free_moves()
        if not moves:
            self._update_time(start_time)
            return [0, 0]  # No valid moves (should not happen in a normal game)
        
        # Evaluate moves with edge control strategy
        best_move = self._edge_control_strategy(moves, start_time)
        self.set_hex(self.player_number, best_move)
        self._update_time(start_time)
        return best_move
        
    def _update_time(self, start_time):
        """Update the time used for this move"""
        move_time = time.time() - start_time
        self.time_used += move_time
        
    def _opening_strategy(self):
        """Special strategy for opening moves"""
        # If player 1 (top-bottom), prefer center or near-center positions
        if self.player_number == 1:
            # If board is odd-sized, take exact center
            if self.size % 2 == 1:
                return [self.center, self.center]
            # If even-sized, take slightly off-center position
            else:
                return [self.center, self.center - 1]
        # If player 2 (left-right), respond to player 1's move
        else:
            # Find player 1's move
            p1_move = None
            for x in range(self.size):
                for y in range(self.size):
                    if self.get_hex([x, y]) == self.adv_number:
                        p1_move = [x, y]
                        break
                if p1_move:
                    break
            
            # If player 1 took center, take adjacent position in our favor
            if p1_move and p1_move[0] == self.center and p1_move[1] == self.center:
                return [self.center + 1, self.center - 1]
            # Otherwise take center if available
            elif self.get_hex([self.center, self.center]) == 0:
                return [self.center, self.center]
            # Otherwise take another strategic position
            else:
                return [self.center - 1, self.center]
                
    def _find_winning_move(self):
        """Find a move that immediately wins the game"""
        for move in self.free_moves():
            board_copy = self.copy()
            board_copy.set_hex(self.player_number, move)
            if board_copy.check_win(self.player_number):
                return move
        return None
        
    def _find_blocking_move(self):
        """Find a move that blocks opponent from winning in their next move"""
        for move in self.free_moves():
            board_copy = self.copy()
            board_copy.set_hex(self.adv_number, move)
            if board_copy.check_win(self.adv_number):
                return move
        return None
        
    def _edge_control_strategy(self, moves, start_time):
        """Main strategy focusing on edge control"""
        move_scores = {}
        total_moves = len(moves)
        
        # Time management - how much time to spend on move evaluation
        time_per_move = min(0.5, (self.time_limit - self.time_used) / 30)
        
        # Categorize moves by their strategic location
        edge_moves = []
        near_edge_moves = []
        central_moves = []
        
        for move in moves:
            x, y = move
            # Edge moves
            if x == 0 or x == self.size - 1 or y == 0 or y == self.size - 1:
                edge_moves.append(move)
            # Near edge moves (one cell away from edge)
            elif x == 1 or x == self.size - 2 or y == 1 or y == self.size - 2:
                near_edge_moves.append(move)
            # Central moves
            else:
                central_moves.append(move)
                
        # Prioritize evaluation - first edges, then near edges, then central
        priority_moves = edge_moves + near_edge_moves + central_moves
        
        # If we have too many moves to evaluate with given time, focus on higher priority moves
        max_moves_to_evaluate = max(10, int((self.time_limit - self.time_used) / time_per_move))
        moves_to_evaluate = priority_moves[:max_moves_to_evaluate]
        
        # Evaluate each move
        for move in moves_to_evaluate:
            # Check time constraints
            if time.time() - start_time > 1.0:
                break
                
            score = self._evaluate_move(move)
            move_scores[tuple(move)] = score
            
        # Return the best move
        if move_scores:
            best_move = max(move_scores.items(), key=lambda x: x[1])[0]
            return [best_move[0], best_move[1]]
        else:
            # Fallback if we couldn't evaluate any moves
            return priority_moves[0]
    
    def _evaluate_move(self, move):
        """Evaluate a potential move with focus on edge control"""
        score = 0
        board_copy = self.copy()
        board_copy.set_hex(self.player_number, move)
        
        # === Edge Control Evaluation ===
        # Higher value for moves on our edges
        if self.player_number == 1:  # top-bottom player
            if move[1] == 0 or move[1] == self.size - 1:
                score += 30
        else:  # left-right player
            if move[0] == 0 or move[0] == self.size - 1:
                score += 30
                
        # Value to "steal" opponent's edges
        if self.player_number == 1:  # top-bottom player
            if move[0] == 0 or move[0] == self.size - 1:
                score += 20
        else:  # left-right player
            if move[1] == 0 or move[1] == self.size - 1:
                score += 20
                
        # Penalty for moves that are far from both edges
        edge_distance = self._distance_to_edge(move)
        score -= edge_distance * 2
                
        # === Connectivity Evaluation ===
        # Value for connecting to existing stones
        connected_stones = 0
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.player_number:
                connected_stones += 1
                # Extra value for connecting to edges
                if self._is_edge_point(neighbor):
                    score += 10
        
        score += connected_stones * 5
        
        # === Path Building Evaluation ===
        # Value for creating/extending a path toward goal
        path_value = self._evaluate_path_building(board_copy)
        score += path_value
        
        # === Strategic Position Evaluation ===
        # Value for positions that control multiple paths
        strategic_value = self._evaluate_strategic_position(move)
        score += strategic_value
        
        # === Opponent Blocking Evaluation ===
        # Value for moves that block opponent's paths
        blocking_value = self._evaluate_blocking(move)
        score += blocking_value
        
        return score
        
    def _distance_to_edge(self, move):
        """Calculate distance to relevant edges"""
        # Cache check
        if tuple(move) in self.edge_distance_cache:
            return self.edge_distance_cache[tuple(move)]
            
        x, y = move
        
        if self.player_number == 1:  # top-bottom player
            # Distance to top or bottom edge
            distance = min(y, self.size - 1 - y)
        else:  # left-right player
            # Distance to left or right edge
            distance = min(x, self.size - 1 - x)
            
        # Cache result
        self.edge_distance_cache[tuple(move)] = distance
        return distance
        
    def _is_edge_point(self, point):
        """Check if a point is on an edge"""
        x, y = point
        return x == 0 or x == self.size - 1 or y == 0 or y == self.size - 1
        
    def _evaluate_path_building(self, board):
        """Evaluate how a move contributes to path building"""
        # Check if we have a shorter path to goal after this move
        my_path = self._shortest_path_length(board, self.player_number)
        
        # If path exists, value inverse to length (shorter is better)
        if my_path < float('inf'):
            return 100 / (my_path + 1)
        
        return 0
        
    def _shortest_path_length(self, board, player):
        """Calculate shortest path length for a player"""
        # Create a unique board hash
        board_hash = hash(str(board._grid.tobytes()))
        cache_key = (board_hash, player)
        
        # Check cache
        if cache_key in self.shortest_path_cache:
            return self.shortest_path_cache[cache_key]
            
        # Player-specific start and end points
        if player == 1:  # top-bottom
            # Start from top edge
            starts = [[x, 0] for x in range(self.size) if board.get_hex([x, 0]) == player]
            # Target is bottom edge
            target_axis = 1
            target_value = self.size - 1
        else:  # left-right
            # Start from left edge
            starts = [[0, y] for y in range(self.size) if board.get_hex([0, y]) == player]
            # Target is right edge
            target_axis = 0
            target_value = self.size - 1
            
        if not starts:
            # No starting points available
            self.shortest_path_cache[cache_key] = float('inf')
            return float('inf')
            
        # Find shortest path using BFS
        min_path = float('inf')
        for start in starts:
            path_length = self._bfs_shortest_path(board, start, player, target_axis, target_value)
            min_path = min(min_path, path_length)
            
        # Cache result
        self.shortest_path_cache[cache_key] = min_path
        return min_path
        
    def _bfs_shortest_path(self, board, start, player, target_axis, target_value):
        """BFS to find shortest path length"""
        from collections import deque
        
        queue = deque([(start, 0)])  # (position, path_length)
        visited = set([tuple(start)])
        
        while queue:
            (x, y), path_length = queue.popleft()
            
            # Check if we've reached target edge
            if (target_axis == 0 and x == target_value) or (target_axis == 1 and y == target_value):
                return path_length
                
            # Explore neighbors
            for neighbor in board.neighbors([x, y]):
                if board.get_hex(neighbor) == player and tuple(neighbor) not in visited:
                    visited.add(tuple(neighbor))
                    queue.append((neighbor, path_length + 1))
                    
        return float('inf')  # No path found
        
    def _evaluate_strategic_position(self, move):
        """Evaluate position based on strategic value"""
        score = 0
        x, y = move
        
        # Center control bonus
        distance_to_center = max(abs(x - self.center), abs(y - self.center))
        center_control = max(0, self.size // 2 - distance_to_center)
        score += center_control * 2
        
        # Bridge creation potential - look for empty cells that would create a bridge
        bridge_potential = 0
        for direction1 in range(6):
            direction2 = (direction1 + 1) % 6  # Adjacent direction
            
            # Get coordinates in these directions
            dx1, dy1 = self._get_direction_offset(direction1)
            dx2, dy2 = self._get_direction_offset(direction2)
            
            pos1 = [x + dx1, y + dy1]
            pos2 = [x + dx2, y + dy2]
            
            # Check if both positions are within bounds
            if (0 <= pos1[0] < self.size and 0 <= pos1[1] < self.size and
                0 <= pos2[0] < self.size and 0 <= pos2[1] < self.size):
                # Check if positions are empty (potential for bridge)
                if self.get_hex(pos1) == 0 and self.get_hex(pos2) == 0:
                    bridge_potential += 1
                    
        score += bridge_potential * 3
        
        # Value for positions that help facilitate both cardinal directions
        # For player 1 (top-bottom), value positions that help east-west movement too
        if self.player_number == 1:
            ew_connection_value = self._evaluate_secondary_axis_connection(move, 0)
            score += ew_connection_value
        # For player 2 (left-right), value positions that help north-south movement too
        else:
            ns_connection_value = self._evaluate_secondary_axis_connection(move, 1)
            score += ns_connection_value
            
        return score
        
    def _get_direction_offset(self, direction):
        """Get coordinate offsets for the 6 hex directions"""
        directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
        return directions[direction]
        
    def _evaluate_secondary_axis_connection(self, move, axis):
        """Evaluate how well a move connects on the secondary axis"""
        value = 0
        x, y = move
        
        # For primary connections, we obviously want to connect along our axis
        # But having some secondary axis connections provides flexibility
        board_copy = self.copy()
        board_copy.set_hex(self.player_number, move)
        
        # Get neighbors along the specified axis
        neighbors = []
        if axis == 0:  # East-west axis
            # Look at neighbors in west direction
            west_neighbors = []
            curr_x, curr_y = x, y
            while curr_x > 0:
                curr_x -= 1
                if [curr_x, curr_y] in self.neighbors([curr_x+1, curr_y]):
                    west_neighbors.append([curr_x, curr_y])
                    break
                    
            # Look at neighbors in east direction
            east_neighbors = []
            curr_x, curr_y = x, y
            while curr_x < self.size - 1:
                curr_x += 1
                if [curr_x, curr_y] in self.neighbors([curr_x-1, curr_y]):
                    east_neighbors.append([curr_x, curr_y])
                    break
                    
            neighbors = west_neighbors + east_neighbors
            
        else:  # North-south axis
            # Look at neighbors in north direction
            north_neighbors = []
            curr_x, curr_y = x, y
            while curr_y > 0:
                curr_y -= 1
                if [curr_x, curr_y] in self.neighbors([curr_x, curr_y+1]):
                    north_neighbors.append([curr_x, curr_y])
                    break
                    
            # Look at neighbors in south direction
            south_neighbors = []
            curr_x, curr_y = x, y
            while curr_y < self.size - 1:
                curr_y += 1
                if [curr_x, curr_y] in self.neighbors([curr_x, curr_y-1]):
                    south_neighbors.append([curr_x, curr_y])
                    break
                    
            neighbors = north_neighbors + south_neighbors
            
        # Check if these neighbors are our stones
        friendly_connections = 0
        for neighbor in neighbors:
            if board_copy.get_hex(neighbor) == self.player_number:
                friendly_connections += 1
                
        # Value connecting on secondary axis
        value += friendly_connections * 4
        
        return value
        
    def _evaluate_blocking(self, move):
        """Evaluate how well a move blocks opponent's progress"""
        value = 0
        
        # Create temporary board with our move
        board_with_our_move = self.copy()
        board_with_our_move.set_hex(self.player_number, move)
        
        # Create temporary board with opponent's move
        board_with_opp_move = self.copy()
        board_with_opp_move.set_hex(self.adv_number, move)
        
        # Calculate opponent's shortest path before and after our move
        opp_path_before = self._shortest_path_length(self, self.adv_number)
        opp_path_after = self._shortest_path_length(board_with_our_move, self.adv_number)
        
        # Value increasing opponent's path length
        if opp_path_after > opp_path_before:
            value += 15 * (opp_path_after - opp_path_before)
            
        # Special bonus for cutting off opponent's path entirely
        if opp_path_before < float('inf') and opp_path_after == float('inf'):
            value += 50
            
        # Value for taking positions that would be good for opponent
        # Check how the move would impact our path if opponent took it
        our_path_before = self._shortest_path_length(self, self.player_number)
        our_path_without_move = self._shortest_path_length(board_with_opp_move, self.player_number)
        
        # If opponent taking this position would hinder us greatly, it's valuable to block
        if our_path_without_move > our_path_before:
            value += 10 * (our_path_without_move - our_path_before)
            
        # Check if position is adjacent to opponent's pieces
        adjacent_opponents = 0
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.adv_number:
                adjacent_opponents += 1
                
        # Value blocking/surrounding opponent's pieces
        value += adjacent_opponents * 4
        
        return value
        
    def update(self, move_other_player):
        """Update internal state after opponent's move"""
        self.set_hex(self.adv_number, move_other_player)
        
        # Clear caches as board has changed
        self.shortest_path_cache = {}