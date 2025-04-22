from agents.agent import Agent
import numpy as np
import random

class BridgeAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "BridgeBuilder"
        # Direction vectors for analyzing bridge patterns
        self.directions = [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]

    def step(self):
        """Make a move prioritizing bridge formations"""
        # Check for any available moves
        available_moves = self.free_moves()
        if not available_moves:
            return [0, 0]  # Fallback move
        
        # First, look for potential bridge completions
        bridge_completion = self.find_bridge_completion()
        if bridge_completion:
            self.set_hex(self.player_number, bridge_completion)
            return bridge_completion
        
        # Next, look for opportunities to start new bridges
        bridge_start = self.find_bridge_start()
        if bridge_start:
            self.set_hex(self.player_number, bridge_start)
            return bridge_start
        
        # If no good bridge moves, prioritize center and edge positions
        strategic_move = self.find_strategic_position()
        if strategic_move:
            self.set_hex(self.player_number, strategic_move)
            return strategic_move
        
        # Fallback to a random move if nothing else is found
        random_move = random.choice(available_moves)
        self.set_hex(self.player_number, random_move)
        return random_move
    
    def update(self, move_other_player):
        """Update internal state with opponent's move"""
        self.set_hex(self.adv_number, move_other_player)
    
    def find_bridge_completion(self):
        """Find a move that completes a bridge between two of our pieces"""
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == self.player_number:
                    # Check for potential bridge completions around this piece
                    for i, dir1 in enumerate(self.directions):
                        # Get the coordinates two steps away in this direction
                        two_step_x = x + 2 * dir1[0]
                        two_step_y = y + 2 * dir1[1]
                        
                        # Check if the two-step position is valid and has our piece
                        if (0 <= two_step_x < self.size and 
                            0 <= two_step_y < self.size and 
                            self.get_hex([two_step_x, two_step_y]) == self.player_number):
                            
                            # Check if the middle position is empty (bridge potential)
                            middle_x = x + dir1[0]
                            middle_y = y + dir1[1]
                            if (0 <= middle_x < self.size and 
                                0 <= middle_y < self.size and 
                                self.get_hex([middle_x, middle_y]) == 0):
                                
                                # This completes a bridge
                                return [middle_x, middle_y]
        
        return None
    
    def find_bridge_start(self):
        """Find a good position to start a new bridge"""
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == self.player_number:
                    # Look for empty spaces two steps away in each direction
                    for dir1 in self.directions:
                        two_step_x = x + 2 * dir1[0]
                        two_step_y = y + 2 * dir1[1]
                        
                        if (0 <= two_step_x < self.size and 
                            0 <= two_step_y < self.size and 
                            self.get_hex([two_step_x, two_step_y]) == 0):
                            
                            # Check if the middle position is also empty
                            middle_x = x + dir1[0]
                            middle_y = y + dir1[1]
                            if (0 <= middle_x < self.size and 
                                0 <= middle_y < self.size and 
                                self.get_hex([middle_x, middle_y]) == 0):
                                
                                # This is a good position to start a bridge
                                return [two_step_x, two_step_y]
        
        return None
    
    def find_strategic_position(self):
        """Find a strategic position (center or edge) based on player number"""
        center = self.size // 2
        available_moves = self.free_moves()
        
        # Convert available_moves to a set for faster lookups
        available_set = {tuple(move) for move in available_moves}
        
        # For Player 1 (horizontal connection), prioritize left-right movement
        if self.player_number == 1:
            # Check center positions first
            for x in range(self.size):
                if tuple([x, center]) in available_set:
                    return [x, center]
            
            # Then check edge positions
            for y in range(self.size):
                if tuple([0, y]) in available_set:  # Left edge
                    return [0, y]
                if tuple([self.size-1, y]) in available_set:  # Right edge
                    return [self.size-1, y]
        
        # For Player 2 (vertical connection), prioritize top-bottom movement
        else:
            # Check center positions first
            for y in range(self.size):
                if tuple([center, y]) in available_set:
                    return [center, y]
            
            # Then check edge positions
            for x in range(self.size):
                if tuple([x, 0]) in available_set:  # Top edge
                    return [x, 0]
                if tuple([x, self.size-1]) in available_set:  # Bottom edge
                    return [x, self.size-1]
        
        # If no strategic position is found, return None
        return None
    
    def evaluate_position(self):
        """Evaluate the current board position based on connectivity"""
        # Simple evaluation - count number of bridges formed
        bridge_count = 0
        
        for x in range(self.size):
            for y in range(self.size):
                if self.get_hex([x, y]) == self.player_number:
                    for i, dir1 in enumerate(self.directions):
                        two_step_x = x + 2 * dir1[0]
                        two_step_y = y + 2 * dir1[1]
                        
                        if (0 <= two_step_x < self.size and 
                            0 <= two_step_y < self.size and 
                            self.get_hex([two_step_x, two_step_y]) == self.player_number):
                            
                            middle_x = x + dir1[0]
                            middle_y = y + dir1[1]
                            if (0 <= middle_x < self.size and 
                                0 <= middle_y < self.size):
                                # Count as bridge even if middle is occupied
                                bridge_count += 1
        
        # Also consider edge connections based on player
        edge_score = 0
        if self.player_number == 1:  # Horizontal player
            # Count pieces on left and right edges
            left_count = sum(1 for y in range(self.size) if self.get_hex([0, y]) == self.player_number)
            right_count = sum(1 for y in range(self.size) if self.get_hex([self.size-1, y]) == self.player_number)
            edge_score = left_count + right_count
        else:  # Vertical player
            # Count pieces on top and bottom edges
            top_count = sum(1 for x in range(self.size) if self.get_hex([x, 0]) == self.player_number)
            bottom_count = sum(1 for x in range(self.size) if self.get_hex([x, self.size-1]) == self.player_number)
            edge_score = top_count + bottom_count
        
        return bridge_count * 2 + edge_score