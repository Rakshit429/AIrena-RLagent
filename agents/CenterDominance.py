import numpy as np
from agents.agent import Agent

class CenterDominanceAgent(Agent):
    def __init__(self, size, player_number, adv_number):
        super().__init__(size, player_number, adv_number)
        self.name = "Center Dominance Agent"
        self._free_spaces = set((x, y) for x in range(self.size) for y in range(self.size))
        # Calculate center of the board
        self.center_x = self.size // 2
        self.center_y = self.size // 2
        
    def step(self):
        # First move: take the center if available
        if len(self._free_spaces) == self.size * self.size:
            move = [self.center_x, self.center_y]
        else:
            # Select the move closest to the center that's still available
            move = self._find_best_move()
            
        # Update internal state
        self._free_spaces.remove((move[0], move[1]))
        self.set_hex(self.player_number, move)
        return move
        
    def update(self, move_other_player):
        self.set_hex(self.adv_number, move_other_player)
        self._free_spaces.remove((move_other_player[0], move_other_player[1]))
        
    def _find_best_move(self):
        """Find the best move based on center dominance strategy."""
        # If we're player 1 (top-bottom), prioritize vertical positioning
        # If we're player 2 (left-right), prioritize horizontal positioning
        
        best_score = float('-inf')
        best_move = None
        
        # Get all free moves
        free_moves = [[x, y] for x, y in self._free_spaces]
        
        for move in free_moves:
            # Calculate base score as negative distance from center
            dx = move[0] - self.center_x
            dy = move[1] - self.center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Base score is inverse of distance (closer to center is better)
            score = -distance
            
            # Bias toward the main connection direction for the player
            if self.player_number == 1:  # top-bottom
                # Prefer moves that are centrally aligned horizontally
                horizontal_bias = -abs(move[0] - self.center_x)
                score += horizontal_bias * 0.5
            else:  # left-right
                # Prefer moves that are centrally aligned vertically
                vertical_bias = -abs(move[1] - self.center_y)
                score += vertical_bias * 0.5
                
            # Check if this move connects to our existing pieces
            has_connection = False
            for neighbor in self.neighbors(move):
                if self.get_hex(neighbor) == self.player_number:
                    has_connection = True
                    score += 1.0
                    break
                    
            # Bonus for moves that block the opponent's path
            blocking_bonus = self._evaluate_blocking_value(move)
            score += blocking_bonus * 1.5
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move
    
    def _evaluate_blocking_value(self, move):
        """Evaluate how well this move blocks the opponent."""
        blocking_value = 0
        
        # Check how many of opponent's pieces would be adjacent
        opponent_neighbors = 0
        for neighbor in self.neighbors(move):
            if self.get_hex(neighbor) == self.adv_number:
                opponent_neighbors += 1
                
        # Higher values for positions that block multiple opponent pieces
        if opponent_neighbors > 1:
            blocking_value += opponent_neighbors * 0.5
            
        return blocking_value