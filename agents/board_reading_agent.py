from agents.agent import Agent
import numpy as np
import heapq
import copy
import random

class BoardReadingAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "BoardReader"
        # Direction vectors for analyzing adjacent hexes
        self.directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]
        
        # Determine the starting and target edges based on player number
        if self.player_number == 1:  # Horizontal player (left to right)
            self.start_edge = "left"
            self.target_edge = "right"
        else:  # Vertical player (top to bottom)
            self.start_edge = "top"
            self.target_edge = "bottom"

    def step(self):
        """Make a move based on board reading and path analysis"""
        # Get available moves
        available_moves = self.free_moves()
        if not available_moves:
            return [0, 0]  # Fallback move
        
        # First priority: Check if we can win in this move
        winning_move = self.find_winning_move()
        if winning_move:
            self.set_hex(self.player_number, winning_move)
            return winning_move
        
        # Second priority: Block opponent's winning move
        blocking_move = self.find_blocking_move()
        if blocking_move:
            self.set_hex(self.player_number, blocking_move)
            return blocking_move
        
        # Third priority: Calculate the best move based on shortest path analysis
        best_move = self.find_shortest_path_move()
        if best_move:
            self.set_hex(self.player_number, best_move)
            return best_move
        
        # Fallback: Choose a strategic position or random move
        strategic_move = self.choose_strategic_move(available_moves)
        self.set_hex(self.player_number, strategic_move)
        return strategic_move
    
    def update(self, move_other_player):
        """Update internal state with opponent's move"""
        self.set_hex(self.adv_number, move_other_player)
    
    def find_winning_move(self):
        """Check if there's a move that would win the game immediately"""
        available_moves = self.free_moves()
        
        for move in available_moves:
            # Make a hypothetical move
            test_board = self.copy()
            test_board.set_hex(self.player_number, move)
            
            # Check if this move would win
            if test_board.check_win(self.player_number):
                return move
        
        return None
    
    def find_blocking_move(self):
        """Check if there's a move that would block opponent's immediate win"""
        available_moves = self.free_moves()
        
        for move in available_moves:
            # Make a hypothetical move for the opponent
            test_board = self.copy()
            test_board.set_hex(self.adv_number, move)
            
            # Check if this move would make the opponent win
            if test_board.check_win(self.adv_number):
                return move
        
        return None
    
    def find_shortest_path_move(self):
        """Find the move that would create the shortest path between edges"""
        available_moves = self.free_moves()
        best_move = None
        best_score = float('inf')
        
        for move in available_moves:
            # Make a hypothetical move
            test_board = self.copy()
            test_board.set_hex(self.player_number, move)
            
            # Calculate shortest path length after this move
            path_length = self.calculate_shortest_path(test_board)
            
            # The lower the path length, the better
            if path_length < best_score:
                best_score = path_length
                best_move = move
        
        return best_move
    
    def calculate_shortest_path(self, board):
        """Calculate the length of the shortest path from start edge to target edge"""
        # Define start and end points based on player number
        if self.player_number == 1:  # Horizontal player
            start_points = [[0, y] for y in range(self.size)]
            end_points = [[self.size-1, y] for y in range(self.size)]
        else:  # Vertical player
            start_points = [[x, 0] for x in range(self.size)]
            end_points = [[x, self.size-1] for x in range(self.size)]
        
        # Filter to include only points that are empty or have our pieces
        start_points = [p for p in start_points if board.get_hex(p) in [0, self.player_number]]
        end_points = [p for p in end_points if board.get_hex(p) in [0, self.player_number]]
        
        # If there are no valid start or end points, return a large value
        if not start_points or not end_points:
            return float('inf')
        
        # Use Dijkstra's algorithm to find shortest path
        min_path = float('inf')
        for start in start_points:
            for end in end_points:
                path_length = self.dijkstra_shortest_path(board, start, end)
                min_path = min(min_path, path_length)
        
        return min_path
    
    def dijkstra_shortest_path(self, board, start, end):
        """Implementation of Dijkstra's algorithm for shortest path"""
        # Convert start and end to tuples for hashability
        start = tuple(start)
        end = tuple(end)
        
        # Initialize distances with infinity
        distances = {(x, y): float('inf') for x in range(self.size) for y in range(self.size)}
        distances[start] = 0
        
        # Priority queue for Dijkstra
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            if current_node == end:
                return current_distance
            
            visited.add(current_node)
            
            # Check all neighbors
            for dir in self.directions:
                nx, ny = current_node[0] + dir[0], current_node[1] + dir[1]
                neighbor = (nx, ny)
                
                # Skip if out of bounds
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                
                # Calculate edge weight based on hex state
                # Empty hex: weight 1
                # Our hex: weight 0
                # Opponent's hex: infinite weight (can't pass through)
                if board.get_hex([nx, ny]) == 0:
                    weight = 1
                elif board.get_hex([nx, ny]) == self.player_number:
                    weight = 0
                else:
                    # Skip opponent's hexes
                    continue
                
                # Update distance if we found a shorter path
                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor))
        
        # If we reach here, there's no path
        return float('inf')
    
    def identify_virtual_connections(self):
        """Identify virtual connections between player's pieces"""
        virtual_connections = []
        
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == self.player_number:
                    # Check for virtual connections in all directions
                    for i, dir1 in enumerate(self.directions):
                        for j, dir2 in enumerate(self.directions[i+1:], i+1):
                            # Get the two adjacent positions
                            pos1 = [x + dir1[0], y + dir1[1]]
                            pos2 = [x + dir2[0], y + dir2[1]]
                            
                            # Check if both positions are in bounds
                            if (0 <= pos1[0] < self.size and 0 <= pos1[1] < self.size and
                                0 <= pos2[0] < self.size and 0 <= pos2[1] < self.size):
                                
                                # Check if one is empty and one has our piece
                                if (self.get_hex(pos1) == 0 and self.get_hex(pos2) == self.player_number):
                                    virtual_connections.append(pos1)
                                elif (self.get_hex(pos1) == self.player_number and self.get_hex(pos2) == 0):
                                    virtual_connections.append(pos2)
        
        return virtual_connections
    
    def choose_strategic_move(self, available_moves):
        """Choose a strategic move based on various heuristics"""
        # Try to find virtual connections first
        virtual_connections = self.identify_virtual_connections()
        valid_virtuals = [vc for vc in virtual_connections if vc in available_moves]
        if valid_virtuals:
            return random.choice(valid_virtuals)
        
        # If no virtual connections, try to play near the center
        center = self.size // 2
        center_distance = {}
        
        for move in available_moves:
            # Calculate Manhattan distance to center
            distance = abs(move[0] - center) + abs(move[1] - center)
            center_distance[tuple(move)] = distance
        
        # Sort moves by distance to center
        sorted_moves = sorted(available_moves, key=lambda m: center_distance[tuple(m)])
        
        # Return one of the top moves (with some randomness)
        top_n = min(3, len(sorted_moves))
        return random.choice(sorted_moves[:top_n])
    
    def evaluate_board(self):
        """Evaluate the current board state"""
        # Combine several evaluation metrics
        
        # 1. Path length (shorter is better)
        path_length = self.calculate_shortest_path(self)
        path_score = 100 - min(100, path_length * 10)  # Lower path length means higher score
        
        # 2. Edge control
        edge_score = 0
        if self.player_number == 1:  # Horizontal player
            for y in range(self.size):
                if self.get_hex([0, y]) == self.player_number:
                    edge_score += 5
                if self.get_hex([self.size-1, y]) == self.player_number:
                    edge_score += 5
        else:  # Vertical player
            for x in range(self.size):
                if self.get_hex([x, 0]) == self.player_number:
                    edge_score += 5
                if self.get_hex([x, self.size-1]) == self.player_number:
                    edge_score += 5
        
        # 3. Virtual connections
        virtual_connection_score = len(self.identify_virtual_connections()) * 3
        
        # Combine scores with weights
        total_score = path_score * 1.5 + edge_score + virtual_connection_score
        
        return total_score